import socket
import threading
import queue
import time
import torch
import logging
import struct
import numpy as np


logger = logging.getLogger("UnityConnection")

class UnityConnection:
    """Handler for Unity connection with binary message format parsing"""
    
    def __init__(self, input_port=8888, output_port=8889, debug_level='INFO'):
        """Initialize Unity connection handler with ports"""
        self.input_port = input_port
        self.output_port = output_port
        self.running = False
        
        # Connection state
        self.input_socket = None
        self.output_socket = None
        self.input_connection = None
        self.output_connection = None
        self.input_client_addr = None
        self.output_client_addr = None
        
        # Communication queues
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()
        
        # Threads
        self.input_thread = None
        self.output_thread = None
        
        # Connection state
        self.connected = threading.Event()
        
        # Frame tracking
        self.frame_counter = 0
        self.frame_timestamps = []
        self.frame_intervals = []
        self.last_frame_time = None

        # Binary format markers
        self.marker_start = b'<START>'
        self.marker_end = b'<END>'
        
    def start(self):
        """Start connection handler for both input and output"""
        if self.running:
            logger.warning("Connection already running")
            return
            
        self.running = True
        self.connected.clear()
        
        # Start the connection listener
        try:
            # Setup input socket
            self.input_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.input_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.input_socket.bind(('0.0.0.0', self.input_port))
            self.input_socket.listen(1)
            
            # Setup output socket
            self.output_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.output_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.output_socket.bind(('0.0.0.0', self.output_port))
            self.output_socket.listen(1)
            
            logger.info(f"Waiting for Unity connections on ports {self.input_port} and {self.output_port}...")
            
            # Start connection acceptor threads
            input_acceptor = threading.Thread(
                target=self._accept_input_connection,
                daemon=True,
                name="InputAcceptor"
            )
            output_acceptor = threading.Thread(
                target=self._accept_output_connection,
                daemon=True,
                name="OutputAcceptor"
            )
            
            input_acceptor.start()
            output_acceptor.start()
            
            return True
            
        except Exception as e:
            logger.error(f"Error starting connection: {e}")
            self.stop()
            return False
    
    def _accept_input_connection(self):
        """Accept incoming connection for input data"""
        try:
            self.input_connection, self.input_client_addr = self.input_socket.accept()
            logger.info(f"Input connection established from {self.input_client_addr}")
            
            # Start input handler thread
            self.input_thread = threading.Thread(
                target=self._handle_input_connection,
                daemon=True,
                name="InputHandler"
            )
            self.input_thread.start()
            
            # Signal connection established
            if self.output_connection:
                self.connected.set()
                
        except Exception as e:
            logger.error(f"Error accepting input connection: {e}")
            self.stop()
    
    def _accept_output_connection(self):
        """Accept incoming connection for output data"""
        try:
            self.output_connection, self.output_client_addr = self.output_socket.accept()
            logger.info(f"Output connection established from {self.output_client_addr}")
            
            # Start output handler thread
            self.output_thread = threading.Thread(
                target=self._handle_output_connection,
                daemon=True,
                name="OutputHandler"
            )
            self.output_thread.start()
            
            # Signal connection established
            if self.input_connection:
                self.connected.set()
                
        except Exception as e:
            logger.error(f"Error accepting output connection: {e}")
            self.stop()
    
    def _handle_input_connection(self):
        """Handle incoming data from Unity with binary message format parsing"""
        logger.info("Starting input connection handler")
        
        total_frames = 0
        frame_rate_check_time = time.time()
        frames_since_check = 0
        
        try:
            # Set socket to non-blocking with timeout for recv operations
            self.input_connection.settimeout(0.5)
            
            while self.running:
                try:
                    # Buffer for collecting data
                    buffer = b''
                    
                    # === STEP 1: Find the start marker ===
                    # Keep reading until we find the "<START>" marker
                    while self.running:
                        chunk = self.input_connection.recv(8192)
                        if not chunk:
                            logger.error("Unity input connection closed (empty chunk)")
                            raise ConnectionError("Connection closed")
                            
                        buffer += chunk
                        
                        # Find start marker
                        start_pos = buffer.find(self.marker_start)
                        if start_pos >= 0:
                            # Remove everything before the start marker
                            buffer = buffer[start_pos + len(self.marker_start):]
                            break
                    
                    # === STEP 2: Read frame header ===
                    # Unity input header structure (18 bytes total; note the
                    # Python -> Unity header is 16 bytes, without FeatureSize):
                    # - MessageLength (uint, 4 bytes) - size of the data portion only
                    # - MessageType (byte, 1 byte) - type of message (1 for input data)
                    # - FrameRate (byte, 1 byte) - target frame rate from Unity
                    # - FrameCounter (ushort, 2 bytes) - frame sequence number
                    # - Timestamp (float, 4 bytes) - Unity timestamp
                    # - DroppedFrames (uint, 4 bytes) - number of dropped frames
                    # - FeatureSize (ushort, 2 bytes) - number of features in the vector
                    header_size = 18
                    
                    # Ensure we have the complete header
                    while len(buffer) < header_size:
                        chunk = self.input_connection.recv(8192)
                        if not chunk:
                            logger.error("Unity input connection closed (empty chunk during header read)")
                            raise ConnectionError("Connection closed")
                        buffer += chunk
                    
                    # Parse header
                    header_data = buffer[:header_size]
                    buffer = buffer[header_size:]
                    
                    # Unpack header with format <IBBHfIH:
                    # < = little endian
                    # I = unsigned int (MessageLength)
                    # B = unsigned char (MessageType)
                    # B = unsigned char (FrameRate)
                    # H = unsigned short (FrameCounter)
                    # f = float (Timestamp)
                    # I = unsigned int (DroppedFrames)
                    # H = unsigned short (FeatureSize)
                    message_length, message_type, frame_rate, frame_counter, timestamp, dropped_frames, feature_size = struct.unpack('<IBBHfIH', header_data)
                    
                    logger.debug(f"Received frame header: message_length={message_length}, "
                            f"message_type={message_type}, frame_rate={frame_rate}, "
                            f"frame_counter={frame_counter}, timestamp={timestamp}, "
                            f"dropped_frames={dropped_frames}")
                    
                    # === STEP 3: Read the frame data ===
                    # Frame data contains:
                    # - Feature vector (feature_size floats from the header, expected 90)
                    # - Head position (3 floats for x, y, z)
                    head_pos_count = 3
                    if feature_size != 90:
                        logger.warning(f"Unexpected feature size from header: {feature_size} (expected 90)")

                    # Calculate total data size in bytes (all floats are 4 bytes each)
                    data_size = (feature_size + head_pos_count) * 4

                    # Ensure we have all the data
                    while len(buffer) < data_size:
                        chunk = self.input_connection.recv(8192)
                        if not chunk:
                            logger.error("Unity input connection closed (empty chunk during data read)")
                            raise ConnectionError("Connection closed")
                        buffer += chunk
                    
                    # Extract data bytes
                    data_bytes = buffer[:data_size]
                    buffer = buffer[data_size:]
                    
                    # === STEP 4: Find and validate the end marker ===
                    end_marker_size = len(self.marker_end)  # "<END>" = 5 bytes
                    
                    # Ensure we have enough data for the end marker
                    while len(buffer) < end_marker_size:
                        chunk = self.input_connection.recv(8192)
                        if not chunk:
                            logger.error("Unity input connection closed (empty chunk during end marker read)")
                            raise ConnectionError("Connection closed")
                        buffer += chunk
                    
                    # Check for end marker
                    end_marker = buffer[:end_marker_size]
                    buffer = buffer[end_marker_size:]
                    
                    if end_marker != self.marker_end:
                        logger.warning(f"Invalid end marker: {end_marker}, expected: {self.marker_end}")
                        # Log some debug info about the received data
                        logger.warning(f"Header size used: {header_size}, Data size: {data_size}")
                        logger.warning(f"Feature size from header: {feature_size}, Head pos count: {head_pos_count}")
                        continue  # Skip this frame and try to recover
                    
                    # === STEP 5: Parse the binary data into numpy arrays ===
                    # Features come first, then head position
                    features_bytes = data_bytes[:feature_size * 4]
                    head_pos_bytes = data_bytes[feature_size * 4:]
                    
                    # Convert bytes to numpy arrays (float32 format)
                    features = np.frombuffer(features_bytes, dtype=np.float32)
                    head_position = np.frombuffer(head_pos_bytes, dtype=np.float32)
                    
                    # Validate array sizes
                    if len(features) != feature_size:
                        logger.error(f"Feature array size mismatch: expected {feature_size}, got {len(features)}")
                        continue
                        
                    if len(head_position) != head_pos_count:
                        logger.error(f"Head position array size mismatch: expected {head_pos_count}, got {len(head_position)}")
                        continue
                    
                    # === STEP 6: Create frame data dictionary ===
                    frame_data = {
                        'features': features,
                        'head_position': head_position,
                        'metadata': {
                            'frame_counter': frame_counter,
                            'timestamp': timestamp,
                            'frame_rate': frame_rate,
                            'dropped_frames': dropped_frames,
                            'feature_size': feature_size,  # Include feature size in metadata
                            'message_type': message_type
                        }
                    }
                    
                    # === STEP 7: Track frame timing and statistics ===
                    current_time = time.time()
                    
                    # First frame handling
                    if self.last_frame_time is None:
                        self.last_frame_time = current_time
                        
                    # Calculate interval and track it
                    interval = current_time - self.last_frame_time
                    self.frame_intervals.append(interval)
                    self.last_frame_time = current_time
                    
                    # Keep only recent intervals (rolling window)
                    if len(self.frame_intervals) > 100:
                        self.frame_intervals = self.frame_intervals[-100:]
                    
                    # Assign a sequential frame counter
                    self.frame_counter += 1
                    total_frames += 1
                    frames_since_check += 1
                    
                    # Add to input queue for processing
                    self.input_queue.put(frame_data)
                    
                    # === STEP 8: Calculate and log frame rate periodically ===
                    if current_time - frame_rate_check_time >= 1.0:
                        measured_fps = frames_since_check / (current_time - frame_rate_check_time)
                        logger.info(f"Received frame {self.frame_counter} | "
                                f"Measured Unity FPS: {measured_fps:.1f} | "
                                f"Frame interval: {interval*1000:.1f}ms | "
                                f"Feature size: {feature_size}D")
                        frame_rate_check_time = current_time
                        frames_since_check = 0
                    
                    # Debug log for first few frames to verify correct parsing
                    if total_frames <= 5:
                        logger.info(f"Frame {frame_counter}: features={features.shape} (size={feature_size}), "
                                f"head_pos={head_position.shape}, timestamp={timestamp:.3f}")
                    
                except (socket.timeout, BlockingIOError):
                    # Normal timeout, just continue
                    continue
                except (ConnectionError, ConnectionResetError) as e:
                    logger.error(f"Connection error: {e}")
                    break
                except struct.error as e:
                    logger.error(f"Binary parsing error: {e}")
                    logger.error(f"Header data length: {len(header_data) if 'header_data' in locals() else 'unknown'}")
                    logger.error(f"Expected header size: {header_size}")
                    # Try to recover by clearing buffer and starting fresh
                    buffer = b''
                    continue
                except Exception as e:
                    logger.error(f"Error processing input data: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    time.sleep(0.1)
                
        except Exception as e:
            logger.error(f"Fatal error in input connection handler: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
        finally:
            logger.info(f"Input connection handler stopping. Total frames processed: {total_frames}")
            self.stop()
    
    def _handle_output_connection(self):
        """Handle outgoing data to Unity"""
        logger.info("Starting output connection handler")
        
        try:
            while self.running:
                try:
                    # Get binary message from queue with timeout
                    binary_message = self.output_queue.get(timeout=0.1)
                    
                    # Send to Unity
                    try:
                        self.output_connection.sendall(binary_message)
                    except Exception as e:
                        logger.error(f"Send error: {e}")
                        break
                        
                except queue.Empty:
                    # No message available, continue waiting
                    continue
                    
        except Exception as e:
            logger.error(f"Error in output connection handler: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
        finally:
            logger.info("Output connection handler stopping")
            self.stop()
    
    def format_binary_message(self, result, metadata=None):
        """
        Format raw prediction data for Unity in a binary format
        Sends predictions including body rotations, root rotation, and beta values
        """
        try:
            # Extract data
            predictions = result.get('predictions', {})
            window = result.get('window', {})
            
            # Get rotation data in 6D format
            body_rot_6d = predictions.get('body_rot', torch.zeros((1, 21, 6)))
            root_rot_6d = predictions.get('root_rot', torch.zeros((1, 6)))
            
            # Get beta values (shape parameters) - new addition
            betas = predictions.get('betas', torch.zeros((1, 10)))
            
            # Get head position
            head_position = window.get('last_head_position', torch.zeros(3))
            
            # Create the message with markers and header
            frame_counter = metadata.get('frame_idx', 0) if metadata else 0
            timestamp = metadata.get('timestamp', time.time()) if metadata else time.time()
            frame_rate = metadata.get('frame_rate', 60) if metadata else 60
            
            # Calculate message length (number of floats * 4 bytes per float)
            root_size = root_rot_6d.numel() * 4      # 6 floats
            body_size = body_rot_6d.numel() * 4      # 21*6 floats
            betas_size = betas.numel() * 4           # 10 floats
            head_pos_size = head_position.numel() * 4 # 3 floats
            message_length = root_size + body_size + betas_size + head_pos_size
            
            # Create header
            header = struct.pack('<IBBHfI',
                message_length,          # uint MessageLength
                2,                       # byte MessageType (2 for result)
                frame_rate,              # byte FrameRate
                frame_counter & 0xFFFF,  # ushort FrameCounter (wrapped to 16 bits)
                timestamp,               # float Timestamp
                0                        # uint DroppedFrames
            )
            
            # Convert data to bytes
            root_rot_bytes = root_rot_6d.cpu().numpy().astype(np.float32).tobytes()
            body_rot_bytes = body_rot_6d.cpu().numpy().astype(np.float32).tobytes()
            betas_bytes = betas.cpu().numpy().astype(np.float32).tobytes()
            head_pos_bytes = head_position.cpu().numpy().astype(np.float32).tobytes()
            
            # Construct full message with markers
            message = self.marker_start + header + root_rot_bytes + body_rot_bytes + betas_bytes + head_pos_bytes + self.marker_end
            
            # Log first few messages
            if self.frame_counter <= 5:
                logger.info(f"Sending message with size: {len(message)} bytes, including {betas.numel()} beta values")
            
            return message
            
        except Exception as e:
            logger.error(f"Error formatting binary message: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def stop(self):
        """Stop all connections and threads"""
        if not self.running:
            return
            
        logger.info("Stopping Unity connections...")
        self.running = False
        self.connected.clear()
        
        # Close connections
        for conn in [self.input_connection, self.output_connection]:
            if conn:
                try:
                    conn.close()
                except:
                    pass
        
        # Close sockets
        for sock in [self.input_socket, self.output_socket]:
            if sock:
                try:
                    sock.close()
                except:
                    pass
        
        # Reset state
        self.input_connection = None
        self.output_connection = None
        self.input_socket = None
        self.output_socket = None
        
        logger.info("Unity connections closed")
    
    def wait_for_connection(self, timeout=60.0):
        """Wait for both connections to be established"""
        return self.connected.wait(timeout=timeout)
    
    def is_connected(self):
        """Check if both connections are established"""
        return self.connected.is_set()
    
    def get_input_queue(self):
        """Get the input queue for incoming data"""
        return self.input_queue
    
    def get_output_queue(self):
        """Get the output queue for outgoing data"""
        return self.output_queue
    
    
    def get_frame_rate_stats(self):
        """Get statistics about incoming frame rate"""
        if not self.frame_intervals:
            return {
                'avg_interval': 0,
                'min_interval': 0,
                'max_interval': 0,
                'estimated_fps': 0
            }
            
        avg_interval = sum(self.frame_intervals) / len(self.frame_intervals)
        min_interval = min(self.frame_intervals)
        max_interval = max(self.frame_intervals)
        
        return {
            'avg_interval': avg_interval,
            'min_interval': min_interval,
            'max_interval': max_interval,
            'estimated_fps': 1.0 / avg_interval if avg_interval > 0 else 0
        }