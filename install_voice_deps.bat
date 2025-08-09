@echo off
echo 🎤 Installing Voice Testing Dependencies for Claude Echo
echo ================================================

echo.
echo 📦 Installing core speech recognition packages...
pip install openai-whisper --upgrade

echo.
echo 🗣️  Installing text-to-speech packages...
pip install pyttsx3 --upgrade

echo.
echo 🎧 Installing audio processing packages...
pip install pyaudio --upgrade

echo.
echo 🧠 Installing machine learning packages for voice learning...
pip install torch torchvision torchaudio --upgrade
pip install scikit-learn --upgrade
pip install resemblyzer --upgrade

echo.
echo 📊 Installing additional dependencies...
pip install librosa --upgrade
pip install soundfile --upgrade
pip install webrtcvad --upgrade

echo.
echo ✅ Voice testing dependencies installation complete!
echo.
echo 🚀 You can now run real voice testing with:
echo    python start_voice_testing.py
echo.
pause