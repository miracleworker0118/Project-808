from RealtimeSTT import AudioToTextRecorder

def process_text(text):
    print(text)

def process_stabilized_text(text):
    print("!!!!")

if __name__ == '__main__':
    recorder = AudioToTextRecorder(
        input_device_index=1,  # Adjust based on your system,
        model="small.en",
        language="en",
        device="cpu",
        spinner=False,
        enable_realtime_transcription=True,
        realtime_model_type="base.en",
        realtime_processing_pause=0.1,
        beam_size_realtime=5,
        on_realtime_transcription_update=process_text,
        on_realtime_transcription_stabilized=process_stabilized_text
    )
    #recorder.text()
    while True:
        recorder.text(process_text)