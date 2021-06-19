import ffmpeg

video = ffmpeg.input('test.MP4')
audio = ffmpeg.input('./assets/bts.m4a')
out = ffmpeg.output(video, audio, 'sync.mp4')

out.run()