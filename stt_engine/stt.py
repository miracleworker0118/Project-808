import threading
import queue
import numpy as np
import pyaudio
import time
import logging
from faster_whisper import WhisperModel
import gc

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class CPUOptimizedSTT:
    def __init__(self):
        """
        CPU-optimized real-time Speech-to-Text engine with Windows loopback support
        Uses PyAudio with WASAPI for proper loopback device access
        """
        # Hard-coded optimal settings for maximum CPU performance
        self.model_size = "small"  # Best for CPU real-time processing
        self.chunk_duration = 6  # Shorter chunks for CPU
        self.overlap_duration = 0.3  # Minimal overlap
        self.vad_threshold = 0.006  # Tuned for CPU processing
        self.sample_rate = 16000
        
        # CPU-only optimization
        self.device = "cpu"
        self.compute_type = "int8"  # Optimal for CPU performance
        
        # Audio settings
        self.channels = 1
        self.chunk_size = 1024  # PyAudio chunk size
        self.format = pyaudio.paFloat32
        
        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()
        
        # Threading setup
        self.audio_queue = queue.Queue(maxsize=8)  # Small queue for low latency
        self.result_queue = queue.Queue(maxsize=4)
        self.is_running = False
        self._stop_event = threading.Event()
        
        # Audio buffer optimization
        self.chunk_samples = int(self.chunk_duration * self.sample_rate)
        self.overlap_samples = int(self.overlap_duration * self.sample_rate)
        self.buffer_size = self.chunk_samples + self.overlap_samples
        
        # Pre-allocated arrays for zero-copy operations
        self.audio_buffer = np.zeros(self.buffer_size, dtype=np.float32)
        self.temp_buffer = np.zeros(self.chunk_samples, dtype=np.float32)
        self.buffer_write_pos = 0
        self.buffer_lock = threading.Lock()
        
        # Load optimized model
        self._load_model()
        
        # Performance stats
        self.stats = {
            'processed': 0,
            'total_time': 0,
            'skipped_silent': 0,
            'avg_latency': 0
        }
        
    def _load_model(self):
        """Load faster-whisper model with CPU optimization"""
        logger.info(f"Loading faster-whisper model: {self.model_size} on {self.device}")
        start_time = time.time()
        
        # faster-whisper with CPU optimization
        self.model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type,
            cpu_threads=6,  # Optimal CPU threads
            num_workers=1,  # Single worker for CPU
            download_root=None,
            local_files_only=False
        )
        
        # Warm up model with dummy audio
        dummy_audio = np.zeros(self.sample_rate, dtype=np.float32)
        list(self.model.transcribe(dummy_audio, beam_size=1, best_of=1))
        
        load_time = time.time() - start_time
        logger.info(f"Model loaded in {load_time:.2f}s")
        
        # Cleanup after loading
        gc.collect()
    
    def _voice_activity_detection(self, audio_chunk):
        """Ultra-fast VAD using RMS energy"""
        rms = np.sqrt(np.mean(audio_chunk ** 2))
        return rms > self.vad_threshold
    
    def list_audio_devices(self):
        """List all available audio devices including loopback devices"""
        print("\n" + "="*60)
        print("AVAILABLE AUDIO DEVICES")
        print("="*60)
        
        device_count = self.audio.get_device_count()
        
        print("\nINPUT DEVICES (including loopback):")
        print("-" * 40)
        input_devices = []
        
        for i in range(device_count):
            try:
                device_info = self.audio.get_device_info_by_index(i)
                if device_info['maxInputChannels'] > 0:
                    input_devices.append((i, device_info))
                    device_name = device_info['name']
                    sample_rate = int(device_info['defaultSampleRate'])
                    
                    # Identify potential loopback devices
                    is_loopback = any(keyword in device_name.lower() for keyword in [
                        'stereo mix', 'what u hear', 'wave out mix', 'loopback', 
                        'speakers', 'cable output', 'voicemeeter', 'virtual'
                    ])
                    
                    marker = " 🔊 [LOOPBACK]" if is_loopback else ""
                    print(f"{i:2}: {device_name} ({sample_rate}Hz){marker}")
                    
            except Exception as e:
                print(f"{i:2}: [Error reading device info]")
        
        print(f"\nFound {len(input_devices)} input devices")
        print("\nLook for devices marked with 🔊 [LOOPBACK] for speaker capture")
        print("Common loopback device names:")
        print("- 'Stereo Mix'")
        print("- 'What U Hear'") 
        print("- 'CABLE Output' (VB-Cable)")
        print("- 'Speakers (loopback)'")
        print("="*60)
    
    def _find_loopback_device(self):
        """Automatically find the best loopback device"""
        device_count = self.audio.get_device_count()
        
        # Priority order for loopback device detection
        loopback_keywords = [
            'stereo mix',
            'what u hear', 
            'cable output',
            'wave out mix',
            'speakers (loopback)',
            'voicemeeter output',
            'virtual cable'
        ]
        
        # First pass: look for exact matches
        for keyword in loopback_keywords:
            for i in range(device_count):
                try:
                    device_info = self.audio.get_device_info_by_index(i)
                    if (device_info['maxInputChannels'] > 0 and 
                        keyword in device_info['name'].lower()):
                        logger.info(f"Found loopback device: {device_info['name']}")
                        return i, device_info
                except:
                    continue
        
        # Second pass: look for any device with loopback indicators
        for i in range(device_count):
            try:
                device_info = self.audio.get_device_info_by_index(i)
                if device_info['maxInputChannels'] > 0:
                    name_lower = device_info['name'].lower()
                    if any(word in name_lower for word in ['loopback', 'mix', 'virtual']):
                        logger.info(f"Found potential loopback device: {device_info['name']}")
                        return i, device_info
            except:
                continue
        
        # Fallback: use default input device
        try:
            default_device = self.audio.get_default_input_device_info()
            logger.warning("No loopback device found, using default input device")
            logger.warning("This will capture microphone, not speaker output")
            logger.warning("Enable 'Stereo Mix' or install VB-Cable for speaker capture")
            return default_device['index'], default_device
        except:
            return None, None
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio callback for real-time audio capture"""
        if status:
            logger.warning(f"Audio status: {status}")
        
        # Convert bytes to numpy array
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        
        # Add to processing queue
        try:
            self.audio_queue.put_nowait(audio_data.copy())
        except queue.Full:
            pass  # Drop frame if queue is full
        
        return (None, pyaudio.paContinue)
    
    def _audio_capture_worker(self, device_index):
        """Audio capture worker using PyAudio stream"""
        try:
            # Open audio stream with loopback device
            stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_callback
            )
            
            logger.info("Loopback audio stream started")
            stream.start_stream()
            
            # Keep stream alive
            while not self._stop_event.is_set() and stream.is_active():
                time.sleep(0.1)
            
            stream.stop_stream()
            stream.close()
            logger.info("Audio stream stopped")
            
        except Exception as e:
            logger.error(f"Audio stream error: {e}")
    
    def _audio_processing_worker(self):
        """Process audio chunks from queue"""
        while not self._stop_event.is_set():
            try:
                # Get audio chunk
                try:
                    audio_chunk = self.audio_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                # Add to buffer
                with self.buffer_lock:
                    space_left = len(self.audio_buffer) - self.buffer_write_pos
                    
                    if space_left >= len(audio_chunk):
                        # Add chunk to buffer
                        self.audio_buffer[self.buffer_write_pos:self.buffer_write_pos + len(audio_chunk)] = audio_chunk
                        self.buffer_write_pos += len(audio_chunk)
                    else:
                        # Process current buffer if we have enough samples
                        if self.buffer_write_pos >= self.chunk_samples:
                            # Copy chunk for transcription
                            np.copyto(self.temp_buffer, self.audio_buffer[:self.chunk_samples])
                            
                            # Voice activity detection
                            if self._voice_activity_detection(self.temp_buffer):
                                try:
                                    # Queue for transcription
                                    transcription_chunk = self.temp_buffer.copy()
                                    threading.Thread(
                                        target=self._transcribe_chunk, 
                                        args=(transcription_chunk,), 
                                        daemon=True
                                    ).start()
                                except Exception as e:
                                    logger.error(f"Transcription thread error: {e}")
                            else:
                                self.stats['skipped_silent'] += 1
                            
                            # Shift buffer (keep overlap)
                            shift_size = self.chunk_samples - self.overlap_samples
                            self.audio_buffer[:self.overlap_samples] = self.audio_buffer[shift_size:self.chunk_samples]
                            self.buffer_write_pos = self.overlap_samples
                        
                        # Add new chunk
                        add_size = min(len(audio_chunk), len(self.audio_buffer) - self.buffer_write_pos)
                        self.audio_buffer[self.buffer_write_pos:self.buffer_write_pos + add_size] = audio_chunk[:add_size]
                        self.buffer_write_pos += add_size
                
                self.audio_queue.task_done()
                
            except Exception as e:
                logger.error(f"Audio processing error: {e}")
    
    def _transcribe_chunk(self, audio_chunk):
        """Transcribe a single audio chunk"""
        try:
            start_time = time.time()
            
            # faster-whisper transcription with maximum speed settings
            segments, info = self.model.transcribe(
                audio_chunk,
                language="en",  # Fixed language for speed
                beam_size=1,    # Minimal beam search for speed
                best_of=1,      # No multiple candidates
                temperature=0,  # Deterministic output
                condition_on_previous_text=False,  # Independent processing
                no_speech_threshold=0.6,
                log_prob_threshold=-1.0,
                compression_ratio_threshold=2.4,
                vad_filter=False,  # We do VAD ourselves
            )
            
            # Extract text
            text_parts = []
            for segment in segments:
                text_parts.append(segment.text.strip())
            
            text = " ".join(text_parts).strip()
            processing_time = time.time() - start_time
            
            # Update stats
            self.stats['processed'] += 1
            self.stats['total_time'] += processing_time
            self.stats['avg_latency'] = self.stats['total_time'] / self.stats['processed']
            
            # Queue result if not empty
            if text and len(text) > 1 and text not in ["", " ", ".", "you", "Thank you."]:
                try:
                    self.result_queue.put_nowait({
                        'text': text,
                        'time': processing_time,
                        'timestamp': time.time()
                    })
                except queue.Full:
                    pass
                    
        except Exception as e:
            logger.error(f"Transcription error: {e}")
    
    def _result_worker(self):
        """Display transcription results"""
        while not self._stop_event.is_set():
            try:
                result = self.result_queue.get(timeout=0.1)
                
                # Format output with performance indicator
                ts = time.strftime("%H:%M:%S", time.localtime(result['timestamp']))
                proc_time = result['time']
                text = result['text']
                
                # Speed indicators
                if proc_time < 0.3:
                    indicator = "⚡"  # Ultra fast
                elif proc_time < 0.6:
                    indicator = "🚀"  # Fast
                elif proc_time < 1.0:
                    indicator = "✅"  # Good
                else:
                    indicator = "⚠️"   # Slow
                
                print(f"[{ts}] {indicator} {text} ({proc_time:.3f}s)")
                
                self.result_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Display error: {e}")
    
    def start(self, device_index=None):
        """Start the CPU-optimized STT engine with loopback capture"""
        if self.is_running:
            return
        
        # Find loopback device
        if device_index is None:
            device_index, device_info = self._find_loopback_device()
        else:
            try:
                device_info = self.audio.get_device_info_by_index(device_index)
            except:
                logger.error(f"Invalid device index: {device_index}")
                return
        
        if device_index is None or device_info is None:
            logger.error("No suitable audio device found!")
            return
        
        logger.info("🎵 Starting CPU-OPTIMIZED STT Engine for SPEAKER OUTPUT...")
        logger.info(f"Model: {self.model_size} | Device: {self.device} | Compute: {self.compute_type}")
        logger.info(f"Audio Device: {device_info['name']}")
        logger.info("Transcribing audio FROM your speakers (what you hear)")
        
        self.is_running = True
        self._stop_event.clear()
        
        # Start worker threads
        threads = [
            threading.Thread(target=self._audio_capture_worker, args=(device_index,), daemon=True),
            threading.Thread(target=self._audio_processing_worker, daemon=True),
            threading.Thread(target=self._result_worker, daemon=True)
        ]
        
        for thread in threads:
            thread.start()
        
        logger.info("Engine running! Press Ctrl+C to stop...")
        logger.info("Play audio on your computer to see transcriptions")
    
    def stop(self):
        """Stop the STT engine"""
        if not self.is_running:
            return
        
        logger.info("Stopping engine...")
        self.is_running = False
        self._stop_event.set()
        
        # Wait for queues
        try:
            self.audio_queue.join()
            self.result_queue.join()
        except:
            pass
        
        # Print final stats
        print(f"\n{'='*60}")
        print(f"CPU-OPTIMIZED STT ENGINE STATISTICS")
        print(f"{'='*60}")
        print(f"Processed chunks: {self.stats['processed']}")
        print(f"Average latency: {self.stats['avg_latency']:.3f}s")
        print(f"Silent chunks skipped: {self.stats['skipped_silent']}")
        print(f"Model: faster-whisper {self.model_size}")
        print(f"Device: CPU ({self.compute_type}, 6 threads)")
        print(f"{'='*60}")
        
        logger.info("Engine stopped")
    
    def __del__(self):
        """Cleanup PyAudio"""
        try:
            self.audio.terminate()
        except:
            pass

def main():
    """Main entry point"""
    engine = CPUOptimizedSTT()
    
    # Uncomment to list available devices (including loopback)
    engine.list_audio_devices()
    print("\nStarting in 3 seconds...")
    time.sleep(3)
    
    try:
        # Start with automatic loopback device detection
        engine.start()
        # Or specify device index: engine.start(device_index=2)
        
        # Keep running
        while engine.is_running:
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        print("\n🛑 Shutdown requested...")
    finally:
        engine.stop()

if __name__ == "__main__":
    main()