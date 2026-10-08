import time
import threading
import logging
import torch
import traceback
from collections import deque


# Get module logger
logger = logging.getLogger("UnityPredictionPipeline")

class UnityPredictionPipeline:
    """
    Real-time prediction pipeline for Unity pose estimation with core performance profiling
    """
    def __init__(self, model, body_model, frame_buffer, unity_connection, config=None):
        """
        Initialize the pipeline
        Args:
            model: Trained pose estimation model
            body_model: SMPL body model
            frame_buffer: Frame buffer for window creation
            unity_connection: Connection to Unity
            config: Configuration dictionary
        """
        # Store components
        self.model = model
        self.body_model = body_model
        self.frame_buffer = frame_buffer
        self.unity_connection = unity_connection
        self.config = config or {}
        
        # Get device
        self.device = next(model.parameters()).device
        
        # Processing state
        self.running = False
        self.input_queue = unity_connection.get_input_queue()
        
        # Threading components
        self.inference_thread = None
        self.inference_ready = threading.Event()
        
        # === CORE PROFILING METRICS ===
        # Model inference timing (pure model forward pass)
        self.model_inference_times = deque(maxlen=100)
        
        # Pipeline timing (end-to-end: frame received to result sent)
        self.pipeline_times = deque(maxlen=100)
        self.pipeline_start_times = {}  # Track start time for each frame
        
        # Performance tracking
        self.inferences_completed = 0
        self.results_sent = 0
        
        # Performance reporting
        self.last_report_time = 0
        self.report_interval = 3.0  # Report every 3 seconds
        self.inferences_since_report = 0
        
        logger.info("Prediction pipeline initialized with core profiling")
    
    def start(self):
        """Start the prediction pipeline"""
        if self.running:
            logger.warning("Pipeline already running")
            return
            
        self.running = True
        
        # Start inference thread
        self.inference_thread = threading.Thread(
            target=self._inference_worker,
            daemon=True,
            name="InferenceThread"
        )
        
        # Signal inference ready
        self.inference_ready.set()
        
        # Reset report timer
        self.last_report_time = time.time()
        
        # Start thread
        self.inference_thread.start()
        
        logger.info("Pipeline thread started")
        
    def stop(self):
        """Stop the pipeline"""
        if not self.running:
            return
            
        logger.info("Stopping pipeline...")
        self.running = False
        
        # Signal thread to stop
        self.inference_ready.set()
        
        # Join thread
        if self.inference_thread and self.inference_thread.is_alive() and self.inference_thread != threading.current_thread():
            try:
                self.inference_thread.join(timeout=2.0)
            except RuntimeError as e:
                logger.error(f"Error in {self.inference_thread.name}: {e}")
        
        # Print final performance summary
        self._print_final_summary()
            
        logger.info("Pipeline stopped")
    
    def _inference_worker(self):
        """
        Worker thread that runs inference with core performance monitoring
        """
        logger.info("Inference worker started")
        
        # Ensure model is in eval mode
        self.model.eval()
        
        while self.running:
            try:
                # Check if it's time to report performance metrics
                current_time = time.time()
                if current_time - self.last_report_time >= self.report_interval:
                    self._report_core_performance()
                    self.last_report_time = current_time
                    self.inferences_since_report = 0
                
                # Wait for signal to start inference
                if not self.inference_ready.wait(timeout=0.001):
                    continue
                    
                # Clear signal so we don't restart immediately
                self.inference_ready.clear()
                
                # Get the latest window for inference
                window = self.frame_buffer.get_inference_window()
                if window is None:
                    # Try again after a short delay
                    time.sleep(0.001)
                    self.inference_ready.set()
                    continue
                
                # === PIPELINE TIMING START ===
                # Track when this frame's processing started
                frame_idx = window.get('frame_idx', 0)
                pipeline_start = time.time()
                
                # === MODEL INFERENCE TIMING ===
                features = window['features']
                
                # Start model timing
                model_start = time.time()
                # print(f"DEBUG: Model start time: {model_start}")
                
                with torch.no_grad():
                    # Run inference
                    predictions = self.model.forward_offline(features)
                    
                # Ensure CUDA sync for accurate timing
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    
                model_end = time.time()
                model_inference_time = model_end - model_start
                # print(f"DEBUG: Model end time: {model_end}, duration: {model_inference_time}")
                
                # Record model timing
                self.model_inference_times.append(model_inference_time)
                
                # Update counters
                self.inferences_completed += 1
                self.inferences_since_report += 1
                
                # Create result data
                result = {
                    'predictions': predictions,
                    'window': window,
                    'frame_idx': frame_idx,
                    'timestamp': window.get('timestamp', time.time()),
                    'inference_idx': self.inferences_completed - 1
                }
                
                # Send to Unity
                self._send_to_unity(result)
                
                # === PIPELINE TIMING END ===
                pipeline_end = time.time()
                pipeline_time = pipeline_end - pipeline_start
                
                # Record pipeline timing
                self.pipeline_times.append(pipeline_time)

                # Signal that we can start the next inference
                self.inference_ready.set()
                
            except Exception as e:
                logger.error(f"Error in inference worker: {e}")
                logger.error(traceback.format_exc())
                time.sleep(0.1)
                
                # Continue working
                self.inference_ready.set()
                
        logger.info(f"Inference worker stopped after {self.inferences_completed} inferences")
    
    def _send_to_unity(self, result):
        """Send raw prediction data to Unity directly"""
        try:
            # Get output queue
            output_queue = self.unity_connection.get_output_queue()
            
            # Create metadata for the message
            metadata = {
                'frame_idx': result.get('frame_idx', 0),
                'timestamp': result.get('timestamp', time.time()),
                'inference_idx': result.get('inference_idx', 0),
                'frame_rate': self.config.get('target_fps', 60)
            }
            
            # Use the connection's format_binary_message method
            binary_message = self.unity_connection.format_binary_message(result, metadata)
            
            # Send to Unity (non-blocking)
            try:
                output_queue.put_nowait(binary_message)
                self.results_sent += 1
            except:
                logger.debug("Output queue full, result discarded")
                
        except Exception as e:
            logger.error(f"Error sending to Unity: {e}")
    
    def _report_core_performance(self):
        """Report the two core performance metrics"""
        if self.inferences_since_report == 0:
            return
            
        # Calculate model FPS
        model_fps = 0
        model_latency = 0
        if self.model_inference_times:
            recent_model_times = list(self.model_inference_times)[-self.inferences_since_report:]
            if recent_model_times:
                avg_model_time = sum(recent_model_times) / len(recent_model_times)
                model_fps = 1.0 / avg_model_time if avg_model_time > 0 else 0
                model_latency = avg_model_time * 1000  # ms
        
        # Calculate pipeline FPS
        pipeline_fps = 0
        pipeline_latency = 0
        if self.pipeline_times:
            recent_pipeline_times = list(self.pipeline_times)[-self.inferences_since_report:]
            if recent_pipeline_times:
                avg_pipeline_time = sum(recent_pipeline_times) / len(recent_pipeline_times)
                pipeline_fps = 1.0 / avg_pipeline_time if avg_pipeline_time > 0 else 0
                pipeline_latency = avg_pipeline_time * 1000  # ms
        
        # Report core metrics
        logger.info(f"CORE METRICS | Model: {model_fps:.1f} Hz ({model_latency:.2f}ms) | "
                   f"Pipeline: {pipeline_fps:.1f} Hz ({pipeline_latency:.2f}ms) | "
                   f"Inferences: {self.inferences_completed}")
    
    def _print_final_summary(self):
        """Print final performance summary"""
        print("\n" + "="*60)
        print("FINAL PERFORMANCE SUMMARY")
        print("="*60)
        
        # Model performance
        if self.model_inference_times:
            avg_model_time = sum(self.model_inference_times) / len(self.model_inference_times)
            model_fps = 1.0 / avg_model_time
            model_latency = avg_model_time * 1000
            min_model_time = min(self.model_inference_times) * 1000
            max_model_time = max(self.model_inference_times) * 1000
            
            print(f"Model Performance:")
            print(f"  Average FPS: {model_fps:.1f} Hz")
            print(f"  Average Latency: {model_latency:.2f}ms")
            print(f"  Min Latency: {min_model_time:.2f}ms")
            print(f"  Max Latency: {max_model_time:.2f}ms")
        
        # Pipeline performance
        if self.pipeline_times:
            avg_pipeline_time = sum(self.pipeline_times) / len(self.pipeline_times)
            pipeline_fps = 1.0 / avg_pipeline_time
            pipeline_latency = avg_pipeline_time * 1000
            min_pipeline_time = min(self.pipeline_times) * 1000
            max_pipeline_time = max(self.pipeline_times) * 1000
            
            print(f"\nPipeline Performance:")
            print(f"  Average FPS: {pipeline_fps:.1f} Hz")
            print(f"  Average Latency: {pipeline_latency:.2f}ms")
            print(f"  Min Latency: {min_pipeline_time:.2f}ms")
            print(f"  Max Latency: {max_pipeline_time:.2f}ms")
        
        # Overall stats
        print(f"\nOverall Statistics:")
        print(f"  Total Inferences: {self.inferences_completed}")
        print(f"  Results Sent: {self.results_sent}")
        
        if self.model_inference_times and self.pipeline_times:
            model_config = self.config.get('model_type', 'unknown').upper()
            seq_len = self.config.get('seq_len', 'unknown')
            print(f"  Model Type: {model_config}")
            print(f"  Sequence Length: {seq_len}")
        
        print("="*60)
    
    def get_current_performance(self):
        """Get current performance metrics for external monitoring"""
        metrics = {
            'model_fps': 0,
            'model_latency_ms': 0,
            'pipeline_fps': 0,
            'pipeline_latency_ms': 0,
            'total_inferences': self.inferences_completed
        }
        
        if self.model_inference_times:
            avg_model_time = sum(self.model_inference_times) / len(self.model_inference_times)
            metrics['model_fps'] = 1.0 / avg_model_time if avg_model_time > 0 else 0
            metrics['model_latency_ms'] = avg_model_time * 1000
            
        if self.pipeline_times:
            avg_pipeline_time = sum(self.pipeline_times) / len(self.pipeline_times)
            metrics['pipeline_fps'] = 1.0 / avg_pipeline_time if avg_pipeline_time > 0 else 0
            metrics['pipeline_latency_ms'] = avg_pipeline_time * 1000
            
        return metrics