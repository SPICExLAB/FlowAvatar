"""FlowAvatar streaming demo.

Receives head/hand tracking from the Unity client over TCP, runs the pose
model, and streams predicted SMPL-X pose parameters back to Unity.

Run from the repository root:

    python -m streaming.live_demo --model gru

Server, model, and feature settings live in configs/streaming.yaml; use
--config to point at an alternative file.
"""

import os
import sys
import time
import threading
import logging
import argparse
import traceback
from pathlib import Path
from queue import Empty

import torch
import yaml

# Allow running both as a module (python -m streaming.live_demo) and as a
# plain script (python streaming/live_demo.py)
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from streaming.connection import UnityConnection
from streaming.frame_buffer import UnityFrameBuffer
from streaming.prediction_pipeline import UnityPredictionPipeline
from streaming.runtime_utils import clean_cuda_memory

from flowavatar.models import load_checkpoint
from flowavatar.config import paths

from human_body_prior.body_model.body_model import BodyModel

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Get module logger
logger = logging.getLogger('FlowAvatarLiveDemo')

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "streaming.yaml"


def load_streaming_config(config_path=DEFAULT_CONFIG_PATH, model_type=None):
    """Load configs/streaming.yaml and flatten it for the selected model.

    Args:
        config_path: Path to the YAML configuration file.
        model_type: Model key under ``models:`` (falls back to
            ``default_model`` from the file).

    Returns:
        Flat dict with server settings, the selected model's settings, and
        ``model_params`` (architecture + feature-masking flags).
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        raw = yaml.safe_load(f)

    models = raw['models']
    model_type = (model_type or raw.get('default_model', 'gru')).lower()
    if model_type not in models:
        raise KeyError(
            f"Model '{model_type}' not found in {config_path} "
            f"(available: {', '.join(models)})")
    model_cfg = models[model_type]

    config = dict(raw.get('server', {}))
    config['model_type'] = model_type
    config['checkpoint'] = model_cfg['checkpoint']
    config['seq_len'] = model_cfg['seq_len']
    config['target_fps'] = model_cfg.get('target_fps', 0)

    # Architecture parameters plus feature-masking flags feed the network
    # constructor. The masking flags must match how the checkpoint was
    # trained (see the comment in configs/streaming.yaml).
    config['model_params'] = {**model_cfg.get('arch', {}),
                              **raw.get('features', {})}
    return config


class LiveDemo:
    """
    Main class for the streaming demo with core performance profiling
    """
    def __init__(self, config):
        """
        Initialize the live demo
        Args:
            config: Flat configuration dict from load_streaming_config()
        """
        self.config = dict(config)
        logger.info(f"Using {self.config['model_type'].upper()} model configuration")

        # Initialize device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        logger.info(f"Model config: {self.config['checkpoint']}, type: {self.config['model_type']}, target FPS: {self.config['target_fps']}")

        # Initialize components
        self.running = False
        self.unity_connection = None
        self.frame_buffer = None
        self.pipeline = None
        self.model = None
        self.body_model = None
        self.reception_thread = None

        # Performance monitoring
        self.performance_monitor_thread = None

    def initialize_model(self):
        """Initialize SMPL-X body model and load pretrained pose model"""
        try:
            logger.info("Initializing body model...")

            if not os.path.exists(paths.smplx_file):
                logger.error(
                    f"SMPL-X model not found at {paths.smplx_file}.\n"
                    "Register at https://smpl-x.is.tue.mpg.de, download the SMPL-X "
                    "model (NPZ) from https://smpl-x.is.tue.mpg.de/download.php, "
                    "and place SMPLX_NEUTRAL.npz at "
                    "body_models/smplx/SMPLX_NEUTRAL.npz (or set the "
                    "SMPLX_MODEL_PATH environment variable).")
                return False

            # Sample GPU memory before model loading
            if torch.cuda.is_available():
                before_memory = torch.cuda.memory_allocated() / (1024**2)  # MB
                logger.info(f"GPU memory before model loading: {before_memory:.1f}MB")

            # Load SMPL-X body model
            self.body_model = BodyModel(
                bm_fname=str(paths.smplx_file),
                num_betas=10,
                model_type='smplx',
                dtype=torch.float32
            ).to(self.device)

            # Load pose estimation model
            logger.info(f"Loading {self.config['model_type'].upper()} pose estimation model...")
            checkpoint_path = os.path.join(paths.checkpoint_dir, self.config['checkpoint'])

            if not os.path.exists(checkpoint_path):
                logger.error(f"Checkpoint not found: {checkpoint_path}")
                raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")

            # Load model using utility function with specified model type
            self.model = load_checkpoint(
                checkpoint_path,
                model_type=self.config.get('model_type', 'gru'),
                model_params=self.config.get('model_params')
            )
            self.model = self.model.to(self.device)
            self.model.eval()

            # Warmup model with dummy inference
            logger.info("Warming up model...")

            input_dim = 90  # All models take the 90-D feature vector
            dummy_features = torch.zeros((1, self.config['seq_len'], input_dim), device=self.device)
            with torch.no_grad():
                _ = self.model.forward_offline(dummy_features)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            logger.info("Model warmup complete")

            # Sample GPU memory after model loading
            if torch.cuda.is_available():
                after_memory = torch.cuda.memory_allocated() / (1024**2)  # MB
                logger.info(f"GPU memory after model loading: {after_memory:.1f}MB")
                logger.info(f"Model loading increased GPU memory by: {after_memory - before_memory:.1f}MB")

            # Print model summary
            if hasattr(self.model, 'parameters'):
                num_params = sum(p.numel() for p in self.model.parameters())
                logger.info(f"Model has {num_params:,} parameters")

            logger.info("Models initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            logger.error(traceback.format_exc())
            return False

    def initialize_components(self):
        """Initialize components for the pipeline"""
        try:
            # Create Unity connection handler
            logger.info("Initializing Unity connection...")
            self.unity_connection = UnityConnection(
                input_port=self.config['input_port'],
                output_port=self.config['output_port'],
                debug_level=self.config['debug_level']
            )

            # Create frame buffer
            logger.info(f"Creating frame buffer with seq_len={self.config['seq_len']}")
            self.frame_buffer = UnityFrameBuffer(
                seq_len=self.config['seq_len'],
                buffer_size=self.config.get('buffer_size', 250),
                device=self.device
            )

            # Create prediction pipeline
            logger.info("Initializing prediction pipeline...")
            self.pipeline = UnityPredictionPipeline(
                model=self.model,
                body_model=self.body_model,
                frame_buffer=self.frame_buffer,
                unity_connection=self.unity_connection,
                config=self.config
            )

            logger.info("Components initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            logger.error(traceback.format_exc())
            return False

    def _reception_loop(self):
        """
        Frame reception loop with continuous operation
        """
        logger.info("Reception loop started")

        frames_received = 0
        try:
            input_queue = self.unity_connection.get_input_queue()

            while self.running and self.unity_connection.is_connected():
                try:
                    # Try to get a frame with minimal timeout
                    frame_data = input_queue.get(timeout=0.001)

                    # Count frames
                    frames_received += 1

                    # Add to frame buffer
                    self.frame_buffer.add_frame(frame_data)

                except Empty:
                    # No frames available, just continue
                    continue
                except Exception as e:
                    logger.error(f"Error in reception loop: {e}")
                    logger.error(traceback.format_exc())
        except Exception as e:
            logger.error(f"Fatal error in reception loop: {e}")
            logger.error(traceback.format_exc())
        finally:
            logger.info(f"Reception loop stopped after receiving {frames_received} frames")

    def _performance_monitor_loop(self):
        """
        Performance monitoring loop that prints core metrics every 10 seconds
        """
        logger.info("Performance monitor started")

        try:
            while self.running:
                time.sleep(10.0)  # Report every 10 seconds

                if not self.running:
                    break

                # Get current performance metrics
                if self.pipeline:
                    metrics = self.pipeline.get_current_performance()

                    # Print focused performance report
                    print(f"\n{'='*50}")
                    print(f"PERFORMANCE CHECKPOINT - {self.config['model_type'].upper()} Model")
                    print(f"{'='*50}")
                    print(f"Model FPS:     {metrics['model_fps']:.1f} Hz ({metrics['model_latency_ms']:.2f}ms)")
                    print(f"Pipeline FPS:  {metrics['pipeline_fps']:.1f} Hz ({metrics['pipeline_latency_ms']:.2f}ms)")
                    print(f"Inferences:    {metrics['total_inferences']}")
                    print(f"{'='*50}\n")

        except Exception as e:
            logger.error(f"Error in performance monitor: {e}")
        finally:
            logger.info("Performance monitor stopped")

    def start(self):
        """Start the demo with proper initialization sequence"""
        if self.running:
            logger.warning("Demo already running")
            return False

        try:
            # Initialize components
            if not self.initialize_model():
                logger.error("Failed to initialize models")
                return False

            if not self.initialize_components():
                logger.error("Failed to initialize components")
                return False

            # Start Unity connection
            logger.info("Starting Unity connection...")
            self.unity_connection.start()

            # Wait for connection
            if not self.unity_connection.wait_for_connection(timeout=30.0):
                logger.error("Timed out waiting for Unity connection")
                self.stop()
                return False

            # Start running
            self.running = True

            # Start reception thread
            self.reception_thread = threading.Thread(
                target=self._reception_loop,
                daemon=True,
                name="ReceptionThread"
            )
            self.reception_thread.start()

            # Start performance monitor thread
            self.performance_monitor_thread = threading.Thread(
                target=self._performance_monitor_loop,
                daemon=True,
                name="PerformanceMonitor"
            )
            self.performance_monitor_thread.start()

            # Start pipeline
            self.pipeline.start()

            logger.info("Demo started successfully")
            print(f"\nRunning {self.config['model_type'].upper()} model with core performance monitoring")
            print("   Model FPS = Pure inference speed")
            print("   Pipeline FPS = End-to-end throughput")
            print("   Performance checkpoints every 10 seconds\n")

            return True

        except Exception as e:
            logger.error(f"Error starting demo: {e}")
            logger.error(traceback.format_exc())
            self.stop()
            return False

    def stop(self):
        """Stop the demo and clean up resources"""
        logger.info("Stopping demo...")
        self.running = False

        # Stop pipeline (this will print the final summary)
        if self.pipeline:
            self.pipeline.stop()

        # Stop Unity connection
        if self.unity_connection:
            self.unity_connection.stop()

        # Wait for reception thread
        if self.reception_thread and self.reception_thread.is_alive():
            self.reception_thread.join(timeout=2.0)

        # Wait for performance monitor thread
        if self.performance_monitor_thread and self.performance_monitor_thread.is_alive():
            self.performance_monitor_thread.join(timeout=2.0)

        # Final memory cleanup
        if torch.cuda.is_available():
            clean_cuda_memory()

        logger.info("Demo stopped")


def main():
    """Main entry point for the FlowAvatar streaming demo"""
    print("\n=== FlowAvatar Streaming Demo ===", flush=True)

    # Parse command-line arguments for model type and config file
    parser = argparse.ArgumentParser(description='FlowAvatar streaming demo')
    parser.add_argument('--model', type=str, default=None,
                        help='Model type to use (default: default_model from the config file)')
    parser.add_argument('--config', type=str, default=str(DEFAULT_CONFIG_PATH),
                        help='Path to the streaming YAML configuration')
    args = parser.parse_args()

    try:
        config = load_streaming_config(args.config, model_type=args.model)
    except (OSError, KeyError, yaml.YAMLError) as e:
        logger.error(f"Failed to load config from {args.config}: {e}")
        return 1

    model_type = config['model_type']

    try:
        # Create and start demo with the selected configuration
        demo = LiveDemo(config)

        if not demo.start():
            logger.error("Failed to start demo")
            return 1

        print(f"\n{model_type.upper()} model running, performance monitoring active...", flush=True)

        # Keep main thread alive
        try:
            while demo.running and demo.unity_connection.is_connected():
                time.sleep(0.1)

        except KeyboardInterrupt:
            print("\n\nStopping demo gracefully...", flush=True)
        finally:
            demo.stop()

    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        logger.error(traceback.format_exc())
        return 1

    print("=== Demo Stopped ===", flush=True)
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.stdout.flush()
    sys.exit(exit_code)
