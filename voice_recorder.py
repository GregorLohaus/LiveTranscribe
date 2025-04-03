import tkinter as tk
import sounddevice as sd
import numpy as np
from threading import Thread
from scipy import signal
from scipy.io import wavfile
from vispy import app, gloo
from vispy.gloo import Texture2D
import sys
from tkinter import filedialog, ttk
import whisper  # Add Whisper import
import queue
import torch
import os
import time
from dataclasses import dataclass
from typing import Optional

# Set the backend to tkinter
app.use_app('tkinter')

# Modern UI Constants
COLORS = {
    'primary': '#2196F3',  # Blue
    'secondary': '#1976D2',  # Darker Blue
    'background': '#F5F5F5',  # Light Gray
    'surface': '#FFFFFF',  # White
    'text': '#212121',  # Dark Gray
    'text_secondary': '#757575',  # Medium Gray
    'border': '#E0E0E0',  # Light Gray
    'success': '#4CAF50',  # Green
    'error': '#F44336',  # Red
}

FONTS = {
    'heading': ('Helvetica', 16, 'bold'),
    'subheading': ('Helvetica', 12, 'bold'),
    'body': ('Helvetica', 10),
    'button': ('Helvetica', 10, 'bold'),
}

class ModernButton(tk.Button):
    def __init__(self, master=None, **kwargs):
        super().__init__(master, **kwargs)
        self.config(
            font=FONTS['button'],
            bg=COLORS['primary'],
            fg='white',
            relief='flat',
            padx=20,
            pady=10,
            cursor='hand2'
        )
        self.bind('<Enter>', lambda e: self.config(bg=COLORS['secondary']))
        self.bind('<Leave>', lambda e: self.config(bg=COLORS['primary']))

class ModernFrame(tk.Frame):
    def __init__(self, master=None, **kwargs):
        super().__init__(master, **kwargs)
        self.config(bg=COLORS['background'])

class ModernLabel(tk.Label):
    def __init__(self, master=None, **kwargs):
        super().__init__(master, **kwargs)
        self.config(
            font=FONTS['body'],
            bg=COLORS['background'],
            fg=COLORS['text']
        )

class ModernText(tk.Text):
    def __init__(self, master=None, **kwargs):
        super().__init__(master, **kwargs)
        self.config(
            font=FONTS['body'],
            bg=COLORS['surface'],
            fg=COLORS['text'],
            relief='flat',
            padx=10,
            pady=10
        )

class ModernScale(tk.Scale):
    def __init__(self, master=None, **kwargs):
        super().__init__(master, **kwargs)
        self.config(
            bg=COLORS['background'],
            fg=COLORS['text'],
            troughcolor=COLORS['primary'],
            activebackground=COLORS['secondary']
        )

class ModernOptionMenu(tk.OptionMenu):
    def __init__(self, master, variable, *values, **kwargs):
        super().__init__(master, variable, *values, **kwargs)
        self.config(
            font=FONTS['body'],
            bg=COLORS['surface'],
            fg=COLORS['text'],
            relief='flat',
            padx=10,
            pady=5
        )

@dataclass
class TranscriptionResult:
    timestamp: float
    text: str
    sequence: int

class SpectrogramCanvas(app.Canvas):
    def __init__(self, parent=None):
        app.Canvas.__init__(self, parent=parent, keys='interactive',
                          size=(400, 200))  # Wider and shorter canvas
        self.program = gloo.Program("""
            #version 120
            attribute vec2 position;
            attribute vec2 texcoord;
            varying vec2 v_texcoord;
            void main()
            {
                gl_Position = vec4(position, 0.0, 1.0);
                v_texcoord = texcoord;
            }
        """, """
            #version 120
            uniform sampler2D texture;
            varying vec2 v_texcoord;
            void main()
            {
                float value = texture2D(texture, v_texcoord).r;
                if (value > 0.0) {
                    // Create a color gradient from blue (low freq) to red (high freq)
                    vec3 color;
                    if (value < 0.3) {
                        // Blue to cyan for lower values
                        color = mix(vec3(0.0, 0.0, 1.0), vec3(0.0, 1.0, 1.0), value * 3.33);
                    } else {
                        // Cyan to red for higher values
                        color = mix(vec3(0.0, 1.0, 1.0), vec3(1.0, 0.0, 0.0), (value - 0.3) * 1.43);
                    }
                    // Make brighter areas more intense
                    color = mix(color, vec3(1.0), value * 0.7);
                    gl_FragColor = vec4(color, 1.0);
                } else {
                    // Black background
                    gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0);
                }
            }
        """)

        # Create texture for spectrogram with a fixed size matching the canvas
        self.texture = Texture2D(np.zeros((200, 400), dtype=np.float32))
        
        # Create vertices for a quad with a border
        border = 0.05  # 5% border
        vertices = np.array([
            # Outer quad (border)
            [-1, -1], [1, -1], [1, 1],
            [-1, -1], [1, 1], [-1, 1],
            # Inner quad (spectrogram)
            [-1+border, -1+border], [1-border, -1+border], [1-border, 1-border],
            [-1+border, -1+border], [1-border, 1-border], [-1+border, 1-border]
        ], dtype=np.float32)
        
        texcoords = np.array([
            # Outer quad texture coordinates
            [0, 0], [1, 0], [1, 1],
            [0, 0], [1, 1], [0, 1],
            # Inner quad texture coordinates
            [0, 0], [1, 0], [1, 1],
            [0, 0], [1, 1], [0, 1]
        ], dtype=np.float32)
        
        # Create a second program for the border
        self.border_program = gloo.Program("""
            #version 120
            attribute vec2 position;
            void main()
            {
                gl_Position = vec4(position, 0.0, 1.0);
            }
        """, """
            #version 120
            void main()
            {
                gl_FragColor = vec4(1.0, 1.0, 1.0, 1.0);  // White border color
            }
        """)
        
        self.program['position'] = vertices[6:]  # Inner quad vertices
        self.program['texcoord'] = texcoords[6:]  # Inner quad texture coordinates
        self.program['texture'] = self.texture
        
        self.border_program['position'] = vertices[:6]  # Outer quad vertices
        
        # Initialize with a black background
        self.background = np.zeros((200, 400), dtype=np.float32)
        self.texture.set_data(self.background)
        
        self.show()
    
    def update_spectrogram(self, data):
        # Create a new texture data array matching the canvas size
        texture_data = np.zeros((200, 400), dtype=np.float32)
        
        # Ensure data is 2D
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
        
        # Get the dimensions of the input data
        height, width = data.shape
        
        # Ensure the data matches the texture size
        if height != 200 or width != 400:
            print(f"Warning: Spectrogram data shape {data.shape} doesn't match texture size (200, 400)")
            return
        
        # Copy the spectrogram data into the texture
        # Flip horizontally to show frequencies from left to right
        data_flipped = np.fliplr(data)
        
        # Update texture data
        self.texture.set_data(data_flipped)
        self.update()
    
    def on_draw(self, event):
        gloo.clear(color='black')
        # Draw border first
        self.border_program.draw('triangles')
        # Draw spectrogram on top
        self.program.draw('triangles')

class VoiceRecorder:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Voice Recorder")
        self.root.geometry("1000x700")  # Larger window for better spacing
        self.root.configure(bg=COLORS['background'])
        
        # Initialize Whisper model with GPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        # Set memory allocation strategy
        if torch.cuda.is_available():
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:512'
            torch.cuda.empty_cache()
        
        # Initialize audio variables
        self.sample_rate = 44100
        self.spectrogram_buffer = []
        self.last_update_time = 0
        self.update_interval = 0.1  # Update spectrogram every 100ms
        self.whisper_buffer = []  # Buffer for Whisper
        self.last_transcription_time = 0
        self.transcription_interval = 5.0  # Process transcription every 5 seconds
        
        # Available Whisper models
        self.whisper_models = {
            "tiny": "tiny",
            "base": "base",
            "small": "small",
            "medium": "medium",
            "large": "large"
        }
        
        # Whisper supported languages
        self.languages = {
            "English": "en",
            "Spanish": "es",
            "French": "fr",
            "German": "de",
            "Italian": "it",
            "Portuguese": "pt",
            "Dutch": "nl",
            "Russian": "ru",
            "Japanese": "ja",
            "Korean": "ko",
            "Chinese": "zh",
            "Arabic": "ar",
            "Hindi": "hi",
            "Turkish": "tr",
            "Polish": "pl",
            "Ukrainian": "uk",
            "Romanian": "ro",
            "Greek": "el",
            "Czech": "cs",
            "Hungarian": "hu",
            "Swedish": "sv",
            "Danish": "da",
            "Finnish": "fi",
            "Norwegian": "no",
            "Hebrew": "he",
            "Indonesian": "id",
            "Malay": "ms",
            "Vietnamese": "vi",
            "Thai": "th"
        }
        
        self.current_model_name = "base"
        self.whisper_model = whisper.load_model(self.whisper_models[self.current_model_name], device=device)
        
        # Create transcription queue and thread
        self.transcription_queue = queue.Queue()
        self.transcription_thread = None
        self.transcription_running = False
        
        # Create result queue for ordered transcription display
        self.result_queue = queue.PriorityQueue()
        self.last_sequence = 0
        self.last_displayed_sequence = 0
        
        # Audio processing settings
        self.chunk_size = 16000 * 2
        self.current_chunk = np.array([], dtype=np.float32)
        self.silence_threshold = 0.01
        self.min_speech_duration = 16000 // 4
        self.max_speech_duration = 16000 * 3
        self.speech_active = False
        self.speech_start = 0
        
        # Create main container with padding
        self.main_container = ModernFrame(self.root)
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Create title
        title_label = ModernLabel(
            self.main_container,
            text="Live Voice Transcription",
            font=FONTS['heading']
        )
        title_label.pack(pady=(0, 20))
        
        # Create left column container for controls
        self.left_container = ModernFrame(self.main_container)
        self.left_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 20))
        
        # Create right column container for transcription
        self.right_container = ModernFrame(self.main_container)
        self.right_container.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Create transcription textbox with label
        transcription_label = ModernLabel(
            self.right_container,
            text="Transcription Output",
            font=FONTS['subheading']
        )
        transcription_label.pack(pady=(0, 10))
        
        # Create a frame for the textbox and scrollbar
        textbox_frame = ModernFrame(self.right_container)
        textbox_frame.pack(fill=tk.BOTH, expand=True)
        
        self.top_textbox = ModernText(textbox_frame, wrap=tk.WORD)
        self.top_textbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Add scrollbar to textbox
        top_scrollbar = ttk.Scrollbar(textbox_frame, orient=tk.VERTICAL, command=self.top_textbox.yview)
        top_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.top_textbox.config(yscrollcommand=top_scrollbar.set)
        
        # Get available audio devices
        self.devices = sd.query_devices()
        self.input_devices = [device for device in self.devices if device['max_input_channels'] > 0]
        self.input_device_names = [f"{device['name']} (ID: {device['index']})" for device in self.input_devices]
        
        # Create control sections with labels
        self._create_device_section()
        self._create_language_section()
        self._create_model_section()
        self._create_audio_settings_section()
        
        # Create start/stop button
        self.toggle_button = ModernButton(
            self.left_container,
            text="Start Recording",
            command=self.toggle_audio
        )
        self.toggle_button.pack(pady=20)
        
        # Create spectrogram section
        spectrogram_label = ModernLabel(
            self.left_container,
            text="Audio Spectrogram",
            font=FONTS['subheading']
        )
        spectrogram_label.pack(pady=(20, 10))
        
        # Create a frame for the spectrogram
        self.spectrogram_frame = ModernFrame(self.left_container, height=200, width=400)
        self.spectrogram_frame.pack(pady=10)
        self.spectrogram_frame.pack_propagate(False)
        
        # Create spectrogram canvas
        self.spectrogram = SpectrogramCanvas(parent=self.spectrogram_frame)
        self.spectrogram.native.pack(fill=tk.BOTH, expand=True)
        
        # Initialize spectrogram data
        self.spectrogram_data = np.zeros((1024, 100), dtype=np.float32)
        
        # Status label
        self.status_label = ModernLabel(
            self.left_container,
            text="Ready",
            font=FONTS['body']
        )
        self.status_label.pack(pady=10)
        
        # Initialize audio stream
        self.stream = None
        self.is_running = False
    
    def _create_device_section(self):
        section_label = ModernLabel(
            self.left_container,
            text="Audio Device",
            font=FONTS['subheading']
        )
        section_label.pack(pady=(10, 5))
        
        self.input_device_var = tk.StringVar()
        self.input_device_dropdown = ModernOptionMenu(
            self.left_container,
            self.input_device_var,
            *self.input_device_names
        )
        self.input_device_dropdown.pack(pady=(0, 10))
        self.input_device_var.set(self.input_device_names[0] if self.input_device_names else "No input devices found")

    def _create_language_section(self):
        section_label = ModernLabel(
            self.left_container,
            text="Language",
            font=FONTS['subheading']
        )
        section_label.pack(pady=(10, 5))
        
        self.language_var = tk.StringVar()
        self.language_dropdown = ModernOptionMenu(
            self.left_container,
            self.language_var,
            *sorted(self.languages.keys())
        )
        self.language_dropdown.pack(pady=(0, 10))
        self.language_var.set("English")

    def _create_model_section(self):
        section_label = ModernLabel(
            self.left_container,
            text="Whisper Model",
            font=FONTS['subheading']
        )
        section_label.pack(pady=(10, 5))
        
        self.model_var = tk.StringVar()
        self.model_dropdown = ModernOptionMenu(
            self.left_container,
            self.model_var,
            *sorted(self.whisper_models.keys())
        )
        self.model_dropdown.pack(pady=(0, 10))
        self.model_var.set("base")

    def _create_audio_settings_section(self):
        section_label = ModernLabel(
            self.left_container,
            text="Audio Settings",
            font=FONTS['subheading']
        )
        section_label.pack(pady=(10, 5))
        
        # Min chunk length slider
        min_chunk_frame = ModernFrame(self.left_container)
        min_chunk_frame.pack(fill=tk.X, pady=(5, 0))
        
        min_chunk_label = ModernLabel(min_chunk_frame, text="Min Chunk Length (s):")
        min_chunk_label.pack(side=tk.LEFT)
        
        self.min_chunk_var = tk.DoubleVar(value=0.25)
        self.min_chunk_slider = ModernScale(
            min_chunk_frame,
            from_=0.1,
            to=1.0,
            resolution=0.1,
            orient=tk.HORIZONTAL,
            variable=self.min_chunk_var,
            command=self._update_min_chunk
        )
        self.min_chunk_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Max chunk length slider
        max_chunk_frame = ModernFrame(self.left_container)
        max_chunk_frame.pack(fill=tk.X, pady=(5, 0))
        
        max_chunk_label = ModernLabel(max_chunk_frame, text="Max Chunk Length (s):")
        max_chunk_label.pack(side=tk.LEFT)
        
        self.max_chunk_var = tk.DoubleVar(value=3.0)
        self.max_chunk_slider = ModernScale(
            max_chunk_frame,
            from_=1.0,
            to=10.0,
            resolution=0.5,
            orient=tk.HORIZONTAL,
            variable=self.max_chunk_var,
            command=self._update_max_chunk
        )
        self.max_chunk_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Silence threshold slider
        silence_frame = ModernFrame(self.left_container)
        silence_frame.pack(fill=tk.X, pady=(5, 0))
        
        silence_label = ModernLabel(silence_frame, text="Silence Threshold:")
        silence_label.pack(side=tk.LEFT)
        
        self.silence_var = tk.DoubleVar(value=0.01)
        self.silence_slider = ModernScale(
            silence_frame,
            from_=0.001,
            to=0.1,
            resolution=0.001,
            orient=tk.HORIZONTAL,
            variable=self.silence_var,
            command=self._update_silence_threshold
        )
        self.silence_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
    
    def _update_silence_threshold(self, value):
        """Update silence threshold when slider changes."""
        self.silence_threshold = float(value)
        print(f"Updated silence threshold to {value}")
    
    def _update_min_chunk(self, value):
        """Update minimum chunk length when slider changes."""
        self.min_speech_duration = int(float(value) * self.sample_rate)
        print(f"Updated min chunk length to {value} seconds ({self.min_speech_duration} samples)")
    
    def _update_max_chunk(self, value):
        """Update maximum chunk length when slider changes."""
        self.max_speech_duration = int(float(value) * self.sample_rate)
        print(f"Updated max chunk length to {value} seconds ({self.max_speech_duration} samples)")
    
    def toggle_audio(self):
        if not self.is_running:
            # Reload Whisper model if model selection changed
            selected_model = self.model_var.get()
            if selected_model != self.current_model_name:
                print(f"Loading Whisper model: {selected_model}")
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self.whisper_model = whisper.load_model(self.whisper_models[selected_model], device=device)
                self.current_model_name = selected_model
            
            self.start_audio()
            self.toggle_button.config(text="Stop")
        else:
            self.stop_audio()
            self.toggle_button.config(text="Start")
    
    def start_audio(self):
        device_id = self.get_selected_device_id()
        if device_id is None:
            self.status_label.config(text="Error: No valid input device selected")
            return
        
        # Start transcription thread
        self.start_transcription_thread()
        
        # Start audio stream
        try:
            self.stream = sd.InputStream(
                device=device_id,
                channels=1,
                samplerate=self.sample_rate,
                callback=self.audio_callback,
                blocksize=4096,
                dtype=np.float32
            )
            self.stream.start()
            self.is_running = True
            self.status_label.config(text="Running...")
        except Exception as e:
            print(f"Error starting audio stream: {str(e)}")
            self.status_label.config(text="Error: Could not start audio")
    
    def stop_audio(self):
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
                self.stream = None
            except Exception as e:
                print(f"Error stopping audio stream: {str(e)}")
        
        self.is_running = False
        
        # Stop transcription thread gracefully
        if self.transcription_thread and self.transcription_thread.is_alive():
            # Clear the queue to prevent the thread from getting stuck
            while not self.transcription_queue.empty():
                try:
                    self.transcription_queue.get_nowait()
                except queue.Empty:
                    break
            
            # Wait for the thread to finish with a timeout
            try:
                self.transcription_thread.join(timeout=1.0)
            except Exception as e:
                print(f"Error joining transcription thread: {str(e)}")
        
        # Clear buffers
        self.spectrogram_buffer = []
        self.whisper_buffer = []
        
        # Clear transcription queue
        while not self.transcription_queue.empty():
            try:
                self.transcription_queue.get_nowait()
            except queue.Empty:
                break
        
        # Update UI
        self.status_label.config(text="Ready")
        self.toggle_button.config(text="Start Recording")
        
        # Clear spectrogram
        self.spectrogram_data = np.zeros((1024, 100), dtype=np.float32)
        self.spectrogram.update_spectrogram(self.spectrogram_data)
    
    def is_speech(self, audio_chunk):
        """Detect if the audio chunk contains speech based on energy."""
        # Ensure audio is in the correct format
        if len(audio_chunk.shape) > 1:
            audio_chunk = audio_chunk.flatten()
        
        # Convert to floating point if needed
        if audio_chunk.dtype != np.float32:
            audio_chunk = audio_chunk.astype(np.float32)
        if np.max(np.abs(audio_chunk)) > 1.0:
            audio_chunk = audio_chunk / 32768.0
        
        # Calculate RMS energy
        energy = np.sqrt(np.mean(audio_chunk**2))
        print(f"Speech detection - Energy: {energy}, Threshold: {self.silence_threshold}")
        return energy > self.silence_threshold

    def audio_callback(self, indata, frames, time, status):
        if status:
            print(f"Audio status: {status}")
        
        try:
            # Process audio data immediately to prevent buffer buildup
            if len(self.spectrogram_buffer) < 2:  # Keep only 2 chunks for spectrogram
                self.spectrogram_buffer.append(indata.copy())
            
            # Add to current chunk
            self.current_chunk = np.append(self.current_chunk, indata.flatten())
            
            # Check if we have enough data for analysis
            if len(self.current_chunk) >= self.min_speech_duration:
                # Check if current segment contains speech
                is_current_speech = self.is_speech(self.current_chunk[-self.min_speech_duration:])
                
                if is_current_speech:
                    if not self.speech_active:
                        # Speech just started
                        self.speech_active = True
                        self.speech_start = len(self.current_chunk) - self.min_speech_duration
                    elif len(self.current_chunk) - self.speech_start >= self.max_speech_duration:
                        # Speech too long, force a chunk
                        chunk = self.current_chunk[self.speech_start:]
                        self.current_chunk = self.current_chunk[-self.min_speech_duration:]
                        self.speech_start = 0
                        self.speech_active = False
                        
                        # Put the chunk in the queue for transcription
                        try:
                            self.transcription_queue.put(chunk, block=False)
                        except queue.Full:
                            print("Transcription queue full, dropping chunk")
                        except Exception as e:
                            print(f"Error queueing audio chunk: {e}")
                else:
                    if self.speech_active:
                        # Speech just ended
                        if len(self.current_chunk) - self.speech_start >= self.min_speech_duration:
                            # Only process if we have enough speech
                            chunk = self.current_chunk[self.speech_start:]
                            self.current_chunk = self.current_chunk[-self.min_speech_duration:]
                            self.speech_start = 0
                            self.speech_active = False
                            
                            # Put the chunk in the queue for transcription
                            try:
                                self.transcription_queue.put(chunk, block=False)
                            except queue.Full:
                                print("Transcription queue full, dropping chunk")
                            except Exception as e:
                                print(f"Error queueing audio chunk: {e}")
                        else:
                            # Not enough speech, discard
                            self.current_chunk = self.current_chunk[-self.min_speech_duration:]
                            self.speech_start = 0
                            self.speech_active = False
            
            # Check if it's time to update the spectrogram
            current_time = time.currentTime
            if current_time - self.last_update_time >= self.update_interval:
                # Process spectrogram if we have enough data
                if len(self.spectrogram_buffer) >= 2:
                    combined_chunk = np.concatenate(self.spectrogram_buffer, axis=0)
                    self.spectrogram_buffer = []  # Clear the buffer
                    self.root.after(0, lambda: self.update_spectrogram(combined_chunk))
                
                self.last_update_time = current_time
        except Exception as e:
            print(f"Error in audio callback: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def get_selected_device_id(self):
        selected_name = self.input_device_var.get()
        for device in self.input_devices:
            if f"{device['name']} (ID: {device['index']})" == selected_name:
                return device['index']
        return None
    
    def update_spectrogram(self, audio_chunk):
        # Get the actual chunk size
        chunk_size = len(audio_chunk)
        # Use a fixed segment size for consistent output
        nperseg = 256  # Fixed segment size
        noverlap = nperseg // 2  # 50% overlap
        
        # Ensure we have enough data
        if chunk_size < nperseg:
            return
        
        # Compute spectrogram of the chunk
        f, t, Sxx = signal.spectrogram(audio_chunk.flatten(), 
                                     fs=self.sample_rate,
                                     nperseg=nperseg,
                                     noverlap=noverlap)
        
        # Convert to dB scale and normalize
        Sxx_db = 10 * np.log10(Sxx + 1e-10)
        
        # Clip values to a reasonable range (-100 to 0 dB)
        Sxx_db = np.clip(Sxx_db, -100, 0)
        
        # Normalize to 0-1 range
        Sxx_db = (Sxx_db + 100) / 100  # Shift and scale to 0-1 range
        
        # Resize the spectrogram to match the texture size
        from scipy.ndimage import zoom
        target_height = 200  # Match the texture height
        target_width = 400   # Match the texture width
        zoom_factors = (target_height / Sxx_db.shape[0], target_width / Sxx_db.shape[1])
        Sxx_db = zoom(Sxx_db, zoom_factors)
        
        # Update spectrogram data
        self.spectrogram_data = Sxx_db.astype(np.float32)
        self.spectrogram.update_spectrogram(self.spectrogram_data)
    
    def start_transcription_thread(self):
        self.transcription_running = True
        self.transcription_thread = Thread(target=self._transcription_worker)
        self.transcription_thread.daemon = True
        self.transcription_thread.start()
    
    def stop_transcription_thread(self):
        self.transcription_running = False
        if self.transcription_thread and self.transcription_thread.is_alive():
            self.transcription_thread.join(timeout=1.0)
    
    def _transcription_worker(self):
        print("Transcription worker started")
        while self.transcription_running:
            try:
                # Get audio data from queue with timeout
                audio_data = self.transcription_queue.get(timeout=0.1)
                if audio_data is not None:
                    print(f"Processing audio chunk of size: {audio_data.shape}")
                    print(f"Audio data type: {audio_data.dtype}, min: {np.min(audio_data)}, max: {np.max(audio_data)}")
                    
                    # Ensure audio data is in the correct format
                    if len(audio_data.shape) > 1:
                        audio_data = audio_data.flatten()
                    
                    # Convert to floating point values in range [-1, 1]
                    if audio_data.dtype != np.float32:
                        audio_data = audio_data.astype(np.float32)
                    if np.max(np.abs(audio_data)) > 1.0:
                        audio_data = audio_data / 32768.0
                    
                    print(f"After conversion - type: {audio_data.dtype}, min: {np.min(audio_data)}, max: {np.max(audio_data)}")
                    
                    # Resample to 16kHz if needed (Whisper's expected sample rate)
                    if self.sample_rate != 16000:
                        print(f"Resampling from {self.sample_rate}Hz to 16kHz")
                        audio_data = signal.resample_poly(audio_data, 16000, self.sample_rate)
                        print(f"After resampling - size: {audio_data.shape}")
                    
                    # Convert to torch tensor and move to GPU
                    audio_data = torch.from_numpy(audio_data)
                    if torch.cuda.is_available():
                        audio_data = audio_data.cuda()
                    
                    # Perform transcription using Whisper
                    print("Starting transcription...")
                    try:
                        # Clear CUDA cache before transcription
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        
                        # Ensure model is on GPU
                        if torch.cuda.is_available():
                            self.whisper_model = self.whisper_model.cuda()
                        
                        # Get selected language code
                        selected_language = self.languages[self.language_var.get()]
                        print(f"Transcribing in language: {selected_language}")
                        
                        # Process the chunk
                        print("Calling Whisper transcribe...")
                        result = self.whisper_model.transcribe(audio_data, language=selected_language)
                        print("Whisper transcribe completed")
                        
                        if result and "text" in result:
                            transcription = result["text"].strip()
                            if transcription:  # Only process non-empty transcriptions
                                print(f"Transcription result: {transcription}")
                                
                                # Create a new transcription result with timestamp and sequence
                                self.last_sequence += 1
                                trans_result = TranscriptionResult(
                                    timestamp=time.time(),
                                    text=transcription,
                                    sequence=self.last_sequence
                                )
                                
                                print(f"Adding result to queue with sequence {self.last_sequence}")
                                # Add to result queue
                                self.result_queue.put((trans_result.sequence, trans_result))
                                
                                # Schedule display update
                                self.root.after(0, self._update_transcription_display)
                            else:
                                print("Empty transcription result, skipping")
                        else:
                            print("No transcription result returned from Whisper")
                        
                        # Clear GPU memory
                        if torch.cuda.is_available():
                            del audio_data
                            torch.cuda.empty_cache()
                    except Exception as e:
                        print(f"Error during transcription: {str(e)}")
                        import traceback
                        traceback.print_exc()
                        # Clear GPU memory on error
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Transcription error: {str(e)}")
                import traceback
                traceback.print_exc()
                # Clear GPU memory on error
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    
    def _update_transcription_display(self):
        """Update the transcription display with results in order."""
        try:
            print(f"Updating transcription display. Queue size: {self.result_queue.qsize()}")
            print(f"Last displayed sequence: {self.last_displayed_sequence}")
            
            while not self.result_queue.empty():
                # Get the next result in sequence
                sequence, result = self.result_queue.get_nowait()
                print(f"Processing sequence {sequence}, text: {result.text}")
                
                # Check if this is the next expected sequence
                if sequence == self.last_displayed_sequence + 1:
                    print(f"Displaying sequence {sequence}")
                    # Update the textbox
                    self.top_textbox.insert(tk.END, result.text + "\n")
                    self.top_textbox.see(tk.END)  # Scroll to the end
                    self.last_displayed_sequence = sequence
                else:
                    print(f"Re-queueing sequence {sequence} (expected {self.last_displayed_sequence + 1})")
                    # Put it back in the queue if it's not the next one
                    self.result_queue.put((sequence, result))
                    break
        except queue.Empty:
            print("Queue is empty")
        except Exception as e:
            print(f"Error updating transcription display: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def _update_transcription_text(self, text):
        """Deprecated - use _update_transcription_display instead"""
        pass
    
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    recorder = VoiceRecorder()
    recorder.run() 