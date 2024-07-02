#TITLE: FINDING & SUMMARIZING EDUCATIONAL YOUTUBE VIDEOS by Rander

#Documentation for youtube-transcript-api link: https://pypi.org/project/youtube-transcript-api/
#Documentation for BART model: https://huggingface.co/docs/transformers/model_doc/bart#transformers.BartForConditionalGeneration
#Guide for using googleapi for Youtube: https://skillshats.com/blogs/how-to-use-the-youtube-api-with-python-a-step-by-step-guide/

# Required libraries
# pip install google-api-python-client
# pip install pandas
# pip install youtube-transcript-api
# pip install transformers
# pip install torch (torch not directly imported in this program because, transformer library uses some functions in torch library)
# pip install isodate

from googleapiclient.discovery import build #library for Youtube interaction
import pandas as pd #library for data display/manipulation
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled #library for transcripting Youtube video
import isodate #library for time duration format
from transformers import BartForConditionalGeneration, BartTokenizer #library to load local AI model in this program

#Base class to handle common YouTube API interactions
class YouTubeAPIClient:
    def __init__(self, api_key):
        self.youtube = build('youtube', 'v3', developerKey=api_key)

    #Find youtube search result from user keyword
    def fetch_keyword_suggestions(self, keyword, max_results=5):
        search_query = f"{keyword} educational"
        search_response = self.youtube.search().list(
            q=search_query,
            type='video',
            part='id,snippet',
            maxResults=max_results
        ).execute()

        data = []
        for search_result in search_response.get('items', []):
            if search_result['id']['kind'] == 'youtube#video':
                data.append({
                    'title': search_result['snippet']['title'],
                    'video_id': search_result['id']['videoId']
                })

        return pd.DataFrame(data)

    #Fetch other details of the youtube video
    def fetch_video_details(self, video_ids):
        video_details_response = self.youtube.videos().list(
            part='contentDetails',
            id=','.join(video_ids)
        ).execute()

        durations = {}
        for video in video_details_response.get('items', []):
            video_id = video['id']
            duration = isodate.parse_duration(video['contentDetails']['duration'])
            durations[video_id] = str(duration)

        return durations


# Derived class to handle video analysis tasks
class YouTubeVideoAnalyzer(YouTubeAPIClient):
    def __init__(self, api_key):
        super().__init__(api_key) #initialize own attribute

        #AI model used is "sshleifer/distilbart-cnn-12-6" which is a variant of BART model
        self.model_name = 'sshleifer/distilbart-cnn-12-6'
        self.model = BartForConditionalGeneration.from_pretrained(self.model_name)
        self.tokenizer = BartTokenizer.from_pretrained(self.model_name)
        print("AI model is ready to be used.")


    #Send the transcript to AI model for summarization
    def summarize(self, text):


        #takes the input text and converts it into tokens that the model can process
        inputs = self.tokenizer.encode("summarize: " + text, return_tensors='pt', max_length=1024, truncation=True)

        #processes the tokenized input to generate output tokens using the AI model
        summary_ids = self.model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)

        #converts the output tokens back into human-readable text.
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        return summary

    #Start transcripting the video and send to AI model for summarization
    def generate_summary(self, video_ids, search_results, video_durations):
        transcripts = {}
        for video_id in video_ids:
            try:
                transcript = YouTubeTranscriptApi.get_transcript(video_id)
                transcript_text = ' '.join([entry['text'] for entry in transcript])
                video_title = search_results.loc[search_results['video_id'] == video_id, 'title'].values[0]
                video_link = f"https://www.youtube.com/watch?v={video_id}"

                #Retrieved the video id and states "Duration unavailable" if it fails to be retrieved
                video_duration = video_durations.get(video_id, 'Duration unavailable')

                #Call the AI summarizer function to summarize video transcript
                summary = self.summarize(transcript_text)

                #Set each video in dictionary to have title, link, duration and its summary
                transcripts[video_id] = {
                    'title': video_title,
                    'link': video_link,
                    'duration': video_duration,
                    'summary': summary
                }
            #If the transcript is not retrieveable, show that the summary is unavailable
            except (NoTranscriptFound, TranscriptsDisabled):
                transcripts[video_id] = {
                    'title': video_title,
                    'link': video_link,
                    'duration': video_duration,
                    'summary': 'Summary unavailable'
                }
            #Display error if fail to retrieve transcript such as slow connection
            except Exception as e:
                print(f"An unexpected error occurred for video {video_id}: {e}")

        return transcripts


if __name__ == "__main__":
    #Introduction


    #Set the API key for youtube search results (This is my API key, you can create your own API key here: https://console.cloud.google.com/)
    API_KEY = 'AIzaSyC-RHYQ2hEL-40qZAlTBCQIYH6rauoZHDE'
    analyzer = YouTubeVideoAnalyzer(API_KEY)

    #Ask user what to search for in youtube
    target_keyword = input("What would you like to search for? :  ")
    search_results = analyzer.fetch_keyword_suggestions(target_keyword)


    # Generate transcripts for the top 5 search results
    top_video_ids = search_results['video_id'].tolist()

    # Fetch video durations
    video_durations = analyzer.fetch_video_details(top_video_ids)

    # Summarize videos
    transcripts = analyzer.generate_summary(top_video_ids, search_results, video_durations)

#------------------------------------------------------------------------------------------------------
#THIS IS ONLY FOR UNIT TESTING
#IT CHECKS IF THE FIRST QUERY IN SEARCH RESULT FOR "heat engine" IS THE SAME AS EXPECTED RESULT IN TERMS OF VIDEO ID (ZjgYWv1-IB0)
#IMPORTANT!!: TOP YOUTUBE VIDEO SEARCH RESULT MAY DIFFER BASED ON LOCATION, SO CHANGE VIDEO ID TEST AND SEARCH QUERY ACCORDINGLY

#REMOVE THE TRIPLE APOSTROPHE BELOW TO START THE UNIT TEST
"""

import unittest
from unittest.mock import patch

class TestYouTubeVideoAnalyzer(unittest.TestCase):
    @patch.object(YouTubeAPIClient, 'fetch_keyword_suggestions')
    @patch.object(YouTubeAPIClient, 'fetch_video_details')
    @patch('youtube_transcript_api.YouTubeTranscriptApi.get_transcript')
    def test_video_id(self, mock_get_transcript, mock_fetch_video_details, mock_fetch_keyword_suggestions):
        # Mock search results
        search_results = pd.DataFrame({
            'title': ['Heat Engines Explained'],
            'video_id': ['ZjgYWv1-IB0']
        })
        mock_fetch_keyword_suggestions.return_value = search_results

        # Mock video details
        video_details = {'ZjgYWv1-IB0': 'PT15M30S'}
        mock_fetch_video_details.return_value = video_details

        api_key = 'AIzaSyC-RHYQ2hEL-40qZAlTBCQIYH6rauoZHDE'
        analyzer = YouTubeVideoAnalyzer(api_key)

        target_keyword = 'heat engine'
        search_results = analyzer.fetch_keyword_suggestions(target_keyword)

        # Validate search results
        self.assertEqual(search_results.iloc[0]['video_id'], 'ZjgYWv1-IB0')

if __name__ == '__main__':
    unittest.main()
    
"""
#REMOVE THE TRIPLE APOSTROPHE ABOVE TO START THE UNIT TEST

#END OF UNIT TEST CODE
