# THIS FILE IS NOT USED IN THE FINAL SOLUTION
from pytube import YouTube # for downloading youtube videos
import datetime
import whisper
import os
import ffmpeg

# This is a workaround for the SSL error that occurs when downloading youtube videos
#import ssl
#ssl._create_default_https_context = ssl._create_unverified_context


def Download(links):
    for cnt, l in enumerate(links):
        video = YouTube(l)
        video = video.streams.get_highest_resolution()
        try:
            video.download("../data/yt_videos/", filename=f"{cnt}.mp4")
        except:
            print(f"An error has occurred in {cnt}th video")


def getYTVideoLinks():
    links = []
    with open("../data/yt_urls.txt", "r") as file:
        for line in file:
            links.append(line)
    return links


def transcribeVideo():
    # list all whisper models
    model = whisper.load_model("large-v3")
    #whisper.DecodingOptions()

    for file in os.listdir("../data/yt_videos/"):
        filename = file.split(".")[0]
        result = model.transcribe(f"../data/yt_videos/{filename}.mp4")
        save_path = f"../data/transcriptions/{filename}.vtt"

        with open(save_path, "w") as file:
            for i, segment in enumerate(result['segments']):
                file.write(str(i + 1) + "\n")
                file.write(str(datetime.timedelta(seconds=segment['start'])) + " --> " + str(datetime.timedelta(seconds=segment['end'])) + "\n")
                file.write(segment['text'].strip() + "\n")
                file.write("\n")


if __name__ == "__main__":
    links = getYTVideoLinks()
    Download(links)
    transcribeVideo()
