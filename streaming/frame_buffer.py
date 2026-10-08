import torch
import numpy as np
import threading
import time
import logging


logger = logging.getLogger("UnityFrameBuffer")

class UnityFrameBuffer:
    """
    Frame buffer for collecting and managing frames from Unity
    with continuous operation and performance monitoring
    """
    def __init__(self, seq_len=80, buffer_size=250, device=None):
        """
        Initialize buffer with capacity
        
        Args:
            seq_len: Sequence length for inference window
            buffer_size: Size of circular buffer
            device: Torch device to use
        """
        self.seq_len = seq_len
        self.buffer_size = buffer_size
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Buffer state
        self.write_ptr = 0        # Where next frame will be written
        self.read_ptr = None      # Most recent frame used for inference
        self.frame_count = 0      # Total frames in buffer (up to buffer_size)
        self.frames_received = 0  # Total frames received (unbounded)
        self.frames_used = 0      # Total frames used for inference
        
        # Lock for thread safety
        self.buffer_lock = threading.RLock()
        
        # Create buffer for all data structures
        self.features_buffer = []  # List of features tensors
        self.head_pos_buffer = []  # List of head position tensors
        self.timestamp_buffer = []  # List of timestamps
        self.frame_idx_buffer = []  # List of frame indices
        
        # Initialize buffer with empty data
        for _ in range(buffer_size):
            self.features_buffer.append(None)
            self.head_pos_buffer.append(None)
            self.timestamp_buffer.append(0.0)
            self.frame_idx_buffer.append(0)
        
        # Window filled flag
        self.window_filled = False
        
        logger.info(f"Initialized frame buffer with seq_len={seq_len}, buffer_size={buffer_size}")
    
    def add_frame(self, frame_data) -> bool:
        """
        Add a frame to the buffer
        
        Args:
            frame_data: Dictionary with frame data
            
        Returns:
            bool: True when window is filled and ready for processing
        """
        try:
            # Extract frame data
            features = frame_data.get('features')
            head_position = frame_data.get('head_position')
            timestamp = frame_data.get('timestamp', time.time())
            frame_counter = frame_data.get('frame_counter', self.frames_received)
            
            # Convert to tensors if needed
            if not isinstance(features, torch.Tensor):
                features = torch.tensor(features, dtype=torch.float32)
            
            if not isinstance(head_position, torch.Tensor) and head_position is not None:
                head_position = torch.tensor(head_position, dtype=torch.float32)
                
            # Acquire lock for thread safety
            with self.buffer_lock:
                # Store data in buffer
                self.features_buffer[self.write_ptr] = features
                self.head_pos_buffer[self.write_ptr] = head_position
                self.timestamp_buffer[self.write_ptr] = timestamp
                self.frame_idx_buffer[self.write_ptr] = frame_counter
                
                # Update counters
                prev_frame_count = self.frame_count
                if self.frame_count < self.buffer_size:
                    self.frame_count += 1
                
                # Log when buffer first fills
                if prev_frame_count < self.seq_len and self.frame_count >= self.seq_len:
                    logger.info(f"Buffer now has {self.frame_count} frames, enough for inference window")
                    self.window_filled = True
                
                # Advance write pointer
                prev_write_ptr = self.write_ptr
                self.write_ptr = (self.write_ptr + 1) % self.buffer_size
                self.frames_received += 1
                
                # Return whether window is ready
                return self.window_filled
                
        except Exception as e:
            logger.error(f"Error adding frame: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def get_inference_window(self):
        """
        Get a window of frames for inference
        
        Returns:
            dict: Dictionary with tensor data for inference or None if not ready
        """
        # Check if we have enough data first
        if not self.window_filled or self.frame_count < self.seq_len:
            return None
        
        with self.buffer_lock:
            try:
                # Start with the most recent frame (one before write pointer)
                start_idx = (self.write_ptr - 1 + self.buffer_size) % self.buffer_size 
                
                # Update read pointer
                self.read_ptr = start_idx
                
                # Collect frames for the window
                features_list = []
                head_pos_list = []
                frames_used = []
                
                # Traverse backwards through the circular buffer
                for i in range(self.seq_len):
                    idx = (start_idx - i + self.buffer_size) % self.buffer_size
                    
                    # Safety check
                    if self.features_buffer[idx] is None:
                        logger.error(f"Found None features at buffer idx {idx}, can't create window")
                        return None
                    
                    # Append frame data (oldest last)
                    features_list.append(self.features_buffer[idx])
                    
                    if self.head_pos_buffer[idx] is not None:
                        head_pos_list.append(self.head_pos_buffer[idx])
                    else:
                        head_pos_list.append(torch.zeros(3, dtype=torch.float32))
                    
                    frames_used.append({
                        'buffer_idx': idx,
                        'frame_idx': self.frame_idx_buffer[idx],
                        'timestamp': self.timestamp_buffer[idx]
                    })
                
                # Reverse lists to have oldest first
                features_list.reverse()
                head_pos_list.reverse()
                frames_used.reverse()
                
                # Convert to tensors for model input
                features_tensor = torch.stack(features_list).unsqueeze(0).to(self.device)
                head_pos_tensor = torch.stack(head_pos_list).unsqueeze(0).to(self.device)
                
                # Last frame's head position
                last_head_position = head_pos_list[-1].to(self.device)
                
                # Update frames used counter
                self.frames_used += self.seq_len
                
                # Create result dictionary
                result = {
                    'features': features_tensor,
                    'head_positions': head_pos_tensor,
                    'last_head_position': last_head_position,
                    'frame_idx': self.frame_idx_buffer[start_idx],
                    'timestamp': self.timestamp_buffer[start_idx]
                }
                
                return result
                
            except Exception as e:
                logger.error(f"Error getting inference window: {e}")
                import traceback
                logger.error(traceback.format_exc())
                return None
    
    def get_stats(self):
        """Get buffer statistics"""
        with self.buffer_lock:
            stats = {
                'frames_received': self.frames_received,
                'frames_used': self.frames_used,
                'buffer_usage': f"{self.frame_count}/{self.buffer_size}",
                'window_filled': self.window_filled,
                'frames_processed_percent': (self.frames_used / max(1, self.frames_received)) * 100
            }
            return stats
    
    def reset(self):
        """Reset buffer state"""
        with self.buffer_lock:
            logger.info("Resetting frame buffer")
            
            # Reset counters and state
            self.write_ptr = 0
            self.read_ptr = None
            self.frame_count = 0
            self.frames_received = 0
            self.frames_used = 0
            self.window_filled = False
            
            # Clear buffer
            for i in range(self.buffer_size):
                self.features_buffer[i] = None
                self.head_pos_buffer[i] = None
                self.timestamp_buffer[i] = 0.0
                self.frame_idx_buffer[i] = 0
            
            logger.info("Buffer state reset")