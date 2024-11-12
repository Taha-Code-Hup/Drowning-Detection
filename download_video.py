import yt_dlp

url = 'https://www.youtube.com/shorts/T-aalGOagbM?feature=share'

ydl_opts = {
    'format': 'bestvideo+bestaudio/best',
    'outtmpl': 'video.%(ext)s',
    'merge_output_format': 'mp4',  # Ensure it outputs a merged mp4 file
}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download([url])
