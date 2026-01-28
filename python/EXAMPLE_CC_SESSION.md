Me

```
Review the README.md file. Install dependencies. Download the 0.6B CustomVoice model (exclude it from git).          
  Start an API server. Make API requests to generate chinese and english audio files (also exclude the WAV files       
  from git). 
```

CC

```
Here's a summary of everything completed:                                                                            
                                                                                                                       
  1. Reviewed README.md — Understood the project: a FastAPI server wrapping Qwen3-TTS behind an OpenAI-compatible      
  /v1/audio/speech endpoint.                                                                                           
  2. Installed dependencies — Ran uv sync, which installed 91 packages (torch, transformers, fastapi, qwen-tts, etc.). 
  3. Downloaded the 0.6B model — Downloaded Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice to                                    
  ./models/Qwen3-TTS-12Hz-0.6B-CustomVoice. The .gitignore already excludes models/ and *.wav, so no changes were      
  needed.                                                                                                              
  4. Started the API server — Launched with CPU settings (QWEN_TTS_DEVICE=cpu, QWEN_TTS_DTYPE=float32,                 
  QWEN_TTS_ATTN=""). Health check confirmed {"status": "ok"}.                                                          
  5. Generated audio files:                                                                                            
    - english_speech.wav (363K) — "Hello, welcome to the Qwen text-to-speech API. This is a demonstration of English   
  speech synthesis."                                                                                                   
    - chinese_speech.wav (442K) — "你好，欢迎使用通义千问文本转语音接口。这是一段中文语音合成的演示。"                 
                                                                                                                       
  Both files were generated using the alloy voice (mapped to Qwen3-TTS speaker "Vivian") in WAV format. CPU inference  
  was slow (~1.5 min per request) but completed successfully. The server has been shut down.
```

Me

```
Next, download the 0.6B base model from here                                                                         
                                                                                                                       
  https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base                                                                 
                                                                                                                       
  The base model allows the user to generate audio based on a reference audio.                                         
                                                                                                                       
  Add a API request parameter, audio_sample, to specify a reference audio file to generate audio. Test the server      
  after you are done.
```

CC

```
Here's a summary of what was done:                                                                                   
                                                                                                                       
  Model downloaded:                                                                                                    
  - Qwen/Qwen3-TTS-12Hz-0.6B-Base saved to ./models/Qwen3-TTS-12Hz-0.6B-Base (already excluded from git via .gitignore)
                                                                                                                       
  Changes to main.py:                                                                                                  
                                                                                                                       
  1. New request parameters on SpeechRequest (main.py:78-79):                                                          
    - audio_sample (optional string) — path, URL, or base64 of a reference audio file for voice cloning                
    - audio_sample_text (optional string) — transcript of the reference audio; when provided, enables ICL (in-context  
  learning) mode for higher quality cloning                                                                            
    - voice now defaults to "alloy" so it's not required when using voice cloning                                      
  2. Base model loading (main.py:220,253-259): The server reads BASE_MODEL_PATH env var. When set, it loads a second   
  base model alongside the custom-voice model.                                                                         
  3. Routing logic (main.py:274-303): When audio_sample is provided, the request is routed to                          
  base_model.generate_voice_clone(). Otherwise, the existing generate_custom_voice() path is used.                     
                                                                                                                       
  All four test scenarios passed (HTTP 200):                                                                           
  - Custom voice (regression) — 156KB                                                                                  
  - Voice cloning, speaker-embedding only (audio_sample alone) — 118KB                                                 
  - Voice cloning, ICL mode (audio_sample + audio_sample_text) — 169KB                                                 
  - Chinese voice cloning with ICL mode — 169KB                                                                        
                                                                                                                       
  To start the server with both models:                                                                                
  MODEL_PATH=./models/Qwen3-TTS-12Hz-0.6B-CustomVoice \                                                                
    BASE_MODEL_PATH=./models/Qwen3-TTS-12Hz-0.6B-Base \                                                                
    QWEN_TTS_DEVICE=cpu \                                                                                              
    QWEN_TTS_DTYPE=float32 \                                                                                           
    QWEN_TTS_ATTN="" \                                                                                                 
    uv run python main.py
```

Me

```
The MODEL_PATH env var should be CUSTOMVOICE_MODEL_PATH
```

CC

```
Done. The env var is now CUSTOMVOICE_MODEL_PATH (default still /model). Server startup becomes:                      
                                                                                                                       
  CUSTOMVOICE_MODEL_PATH=./models/Qwen3-TTS-12Hz-0.6B-CustomVoice \                                                    
    BASE_MODEL_PATH=./models/Qwen3-TTS-12Hz-0.6B-Base \                                                                
    QWEN_TTS_DEVICE=cpu \                                                                                              
    QWEN_TTS_DTYPE=float32 \                                                                                           
    QWEN_TTS_ATTN="" \                                                                                                 
    uv run python main.py
```

Me

```
Please test the case when you only have CUSTOMVOICE_MODEL_PATH or BASE_MODEL_PATH. In those cases, it is okay        
  that only part of the API would work. The unsupported parameters, such as audio_sample would return an error if      
  only the CUSTOMVOICE_MODEL_PATH is loaded. Save the generated wav files so that I can review
```

CC

```
All tests pass. Here's the summary:                                                                                  
                                                                                                                       
  Scenario 1: Only CUSTOMVOICE_MODEL_PATH                                                                              
  ┌──────────────────────┬──────────────────────────────────────────────────────────────────────────────────────────┐  
  │         Test         │                                          Result                                          │  
  ├──────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────┤  
  │ Custom voice request │ 200 OK — saved as customvoice_only_ok.wav (172KB)                                        │  
  ├──────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────┤  
  │ audio_sample request │ 400 — "audio_sample requires a base model. Set BASE_MODEL_PATH to enable voice cloning." │  
  └──────────────────────┴──────────────────────────────────────────────────────────────────────────────────────────┘  
  Scenario 2: Only BASE_MODEL_PATH                                                                                     
  Test: Voice cloning with audio_sample                                                                                
  Result: 200 OK — saved as basemodel_only_ok.wav (150KB)                                                              
  ────────────────────────────────────────                                                                             
  Test: Custom voice request (no audio_sample)                                                                         
  Result: 400 — "Custom-voice model is not loaded. Set CUSTOMVOICE_MODEL_PATH to enable speaker voices, or provide     
    audio_sample to use voice cloning."                                                                                
  The server also raises a RuntimeError at startup if neither env var is set. Four WAV files are saved for your review:
  - english_speech.wav / chinese_speech.wav — from the earlier session                                                 
  - customvoice_only_ok.wav — custom voice with only the CustomVoice model loaded                                      
  - basemodel_only_ok.wav — voice cloning with only the Base model loaded
```

Me

```
update the README with the new parameters. The quickstart example should show that both CUSTOMVOICE and BASE         
  models are loaded and all API endpoints working properly. But add a note to say which API parameters require         
  which model.
```

CC

```
The README is updated. Here's what changed:                                                                          
                                                                                                                       
  - Download models — now lists both model families (CustomVoice and Base) with a note that you can load one or both   
  - Docker quickstart — mounts both models and sets both env vars                                                      
  - API reference table — added audio_sample, audio_sample_text fields and a new "Requires model" column showing which 
  model each parameter needs                                                                                           
  - Note block — explains the routing: audio_sample present uses Base model, absent uses CustomVoice; HTTP 400 if the  
  required model isn't loaded                                                                                          
  - Two curl examples — one for predefined voice, one for voice cloning                                                
  - Running from source — both GPU and CPU examples set both CUSTOMVOICE_MODEL_PATH and BASE_MODEL_PATH                
  - Environment variables — replaced MODEL_PATH with CUSTOMVOICE_MODEL_PATH and BASE_MODEL_PATH, both with -- default  
  (no longer hardcoded to /model)
```

Me

```
Update the GitHub CI script and it goes like this. Upload all generated audio files to GitHUb Action's artifact      
                                                                                                                       
  1 Start the API server with both models.                                                                             
  2 Test voice generation for both English and Chinese for Vivian, and save the audio files                            
  3 Test voice generation for both English and Chinese using the audio files from previous step as audio_sample        
  4 Stop the API server                                                                                                
  5 Start the API server with CUSTOMVOICE model only                                                                   
  6 Test voice generation for both English and Chinese for Ryan                                                        
  7 Test audio_sample and get an error                                                                                 
  8 Start the API server with BASE model only                                                                          
  9 Test voice generation with Ryan as the audo_sample                                                                 
  10 Test voice name and get an error
```


