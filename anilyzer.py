import os
import argparse
import requests
import pandas as pd
import time
import logging
import numpy as np
from tqdm import tqdm
from typing import Dict, List
from dateutil import parser as date_parser
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnimeDataFetcher:
    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
    }
    PROXY_LIST_PATH = "http.txt"

    def __init__(self, user_id: str, disable_proxy: bool = False):
        self.user_id = user_id
        self.disable_proxy = disable_proxy
        self.proxies = self.get_proxy_list()
        self.proxy_index = 0

    def fetch_anime_data(self) -> Dict:
        url = f"https://shikimori.one/api/users/{self.user_id}/anime_rates?limit=500"
        while True:
            response = requests.get(url, headers=self.HEADERS)
            if response.status_code == 200:
                break
            elif response.status_code == 429:
                logger.info("Rate limit exceeded. Retrying after 60 seconds...")
                time.sleep(60)
            else:
                response.raise_for_status()
        data = response.json()
        return data

    def get_proxy_list(self) -> List[Dict[str, str]]:
        proxies = []
        with open(self.PROXY_LIST_PATH, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if line:
                    proxy = {"http": f"http://{line}"}
                    proxies.append(proxy)
        return proxies

    def download_anime_data(self) -> pd.DataFrame:
        data = self.fetch_anime_data()
        df = pd.DataFrame(data)
        df = self.preprocess_dataframe(df)
        df = self.populate_public_data(df)
        df = self.fix_data_types(df)
        df.to_pickle(f"{self.user_id}_anime_data.pkl")
        logger.info(f"Data for user {self.user_id} downloaded and saved as {self.user_id}_anime_data.pkl")
        return df

    def preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df['user_score'] = df['score'].replace(0, pd.NA)
        df['user_created_at'] = pd.to_datetime(df['created_at'])
        df['user_updated_at'] = pd.to_datetime(df['updated_at'])
        df['user_duration'] = (df['user_updated_at'] - df['user_created_at']).dt.total_seconds()
        df['anime_ids'] = df['anime'].apply(lambda x: x['id'])
        return df

    def populate_public_data(self, df: pd.DataFrame) -> pd.DataFrame:
        anime_ids = df['anime_ids']
        with requests.Session() as session:
            for i, anime_id in tqdm(enumerate(anime_ids, start=1), total=len(anime_ids), desc="Downloading data"):
                anime_data = self.fetch_public_data(session, anime_id)
                df.loc[i-1, 'public_score'] = anime_data['score']
                df.loc[i-1, 'public_episodes'] = anime_data['episodes']
                df.loc[i-1, 'public_rating'] = anime_data['rating']
                df.loc[i-1, 'public_duration'] = anime_data['duration']
                df.loc[i-1, 'public_genres'] = ', '.join([genre['name'] for genre in anime_data['genres']])
                df.loc[i-1, 'public_studios'] = ', '.join([studio['name'] for studio in anime_data['studios']])
        df['year'] = df['user_created_at'].dt.year
        df['month'] = df['user_created_at'].dt.month
        return df

    def fetch_public_data(self, session, anime_id) -> Dict:
        url = f"https://shikimori.one/api/animes/{anime_id}"
        rate_limit_exceeded = False
        while True:
            proxy = self.proxies[self.proxy_index]
            try:
                response = session.get(url, headers=self.HEADERS, proxies=(proxy if not self.disable_proxy else None))
                if response.status_code == 200:
                    break
                elif response.status_code == 429:
                    if not rate_limit_exceeded:
                        logger.info("Rate limit exceeded. Switching proxy and retrying...")
                        rate_limit_exceeded = True
                    self.proxy_index = (self.proxy_index + 1) % len(self.proxies)
                else:
                    response.raise_for_status()
            except requests.exceptions.RequestException as e:
                logger.error(f"Request failed with proxy {proxy}: {e}")
                if not self.proxies:
                    raise RuntimeErrorquote("Proxy list exhausted. Unable to make successful request.")
                self.proxy_index = (self.proxy_index + 1) % len(self.proxies)
        return response.json()

    @staticmethod
    def fix_data_types(df: pd.DataFrame) -> pd.DataFrame:
        print(df.columns)
        df['user_score'] = pd.to_numeric(df['user_score'], errors='coerce')
        df['user_created_at'] = pd.to_datetime(df['user_created_at'])
        df['user_updated_at'] = pd.to_datetime(df['user_updated_at'])
        df['user_duration'] = pd.to_numeric(df['user_duration'], errors='coerce')
        df['anime_ids'] = pd.to_numeric(df['anime_ids'], errors='coerce')
        df['public_score'] = pd.to_numeric(df['public_score'], errors='coerce')
        df['public_episodes'] = pd.to_numeric(df['public_episodes'], errors='coerce')
        df['public_duration'] = pd.to_numeric(df['public_duration'], errors='coerce')
        df['year'] = pd.to_numeric(df['year'], errors='coerce')
        df['month'] = pd.to_numeric(df['month'], errors='coerce')
        return df


class AnimeDataVisualizer:
    @staticmethod
    def extract_anime_name(anime: Dict) -> str:
        return anime['name']

    def visualize_data(self, df: pd.DataFrame):
        top_user_scores = df.nlargest(10, 'user_score')[['anime', 'user_score']]
        top_user_scores['anime'] = top_user_scores['anime'].apply(self.extract_anime_name)
        top_public_scores = df.nlargest(10, 'public_score')[['anime', 'public_score']]
        top_public_scores['anime'] = top_public_scores['anime'].apply(self.extract_anime_name)

        fig, axs = plt.subplots(2, 1, figsize=(10, 10))

        sns.barplot(data=top_user_scores, x='user_score', y='anime', ax=axs[0])
        axs[0].set_title('Top 10 Animes by User Score')

        sns.barplot(data=top_public_scores, x='public_score', y='anime', ax=axs[1])
        axs[1].set_title('Top 10 Animes by Public Score')

        plt.tight_layout()
        plt.show()

class AnimeDataAnalyzer:
    def genre_preferences_over_time(self, df: pd.DataFrame):
        df['public_genres'] = df['public_genres'].apply(lambda x: x.split(', '))
        grouped = df.groupby('year')
        genre_frequency_by_year = {}

        for year, group in grouped:
            all_genres = [genre for sublist in group['public_genres'].tolist() for genre in sublist]
            genre_frequency = Counter(all_genres)
            genre_frequency_by_year[year] = genre_frequency

        df_genre_frequency_by_year = pd.DataFrame(genre_frequency_by_year).T.fillna(0)

        plt.figure(figsize=(10, 6))
        for genre in df_genre_frequency_by_year.columns:
            plt.plot(df_genre_frequency_by_year.index, df_genre_frequency_by_year[genre], label=genre)

        plt.title('Genre preferences over time')
        plt.xlabel('Year')
        plt.ylabel('Number of animes')
        plt.legend()
        plt.show()

    def genre_preferences_over_time_dt(self, df: pd.DataFrame, derivative_interval: int = 100):
        from scipy.signal import savgol_filter

        df['date'] = pd.to_datetime(df['user_updated_at']).dt.date
        grouped = df.groupby('date')
        genre_frequency_by_date = {}

        for date, group in grouped:
            all_genres = [genre for sublist in group['public_genres'].tolist() for genre in sublist]
            genre_frequency = Counter(all_genres)
            genre_frequency_by_date[date] = genre_frequency

        df_genre_frequency_by_date = pd.DataFrame(genre_frequency_by_date).T.fillna(0)
        df_genre_frequency_by_date = df_genre_frequency_by_date.sort_index()

        # Calculate the derivative of genre preferences
        num_genres = len(df_genre_frequency_by_date.columns)
        min_window_length = min(derivative_interval, len(df_genre_frequency_by_date))

        print(min_window_length)

        df_genre_frequency_by_date_smoothed = df_genre_frequency_by_date.apply(
            lambda x: savgol_filter(x, window_length=min_window_length, polyorder=3, deriv=1)
        )

        plt.figure(figsize=(10, 6))
        for genre in df_genre_frequency_by_date_smoothed.columns:
            plt.plot(df_genre_frequency_by_date_smoothed.index, df_genre_frequency_by_date_smoothed[genre], label=genre)

        plt.title('Genre Preferences over Time (Derivative)')
        plt.xlabel('Date')
        plt.ylabel('Rate of Change')
        plt.legend()
        plt.show()

        for genre in df_genre_frequency_by_date_smoothed.columns:
            genre_derivative = df_genre_frequency_by_date_smoothed[genre]

            max_derivative_date = genre_derivative.idxmax()
            max_derivative = genre_derivative.max()

            min_derivative_date = genre_derivative.idxmin()
            min_derivative = genre_derivative.min()

            threshold = 0.01  # Adjust the threshold for significance as needed

            if abs(max_derivative) > threshold or abs(min_derivative) > threshold:
                print(f"Genre: {genre}")
                
                if abs(max_derivative) > threshold:
                    print(f"Max Rate of Change: {max_derivative:.2f} on {max_derivative_date}")
                
                if abs(min_derivative) > threshold:
                    print(f"Min Rate of Change: {min_derivative:.2f} on {min_derivative_date}")

                print("--------------------------------------")


    def favorite_studio(self, df: pd.DataFrame):
        df['public_studios'] = df['public_studios'].apply(lambda x: x.split(', '))
        all_studios = [studio for sublist in df['public_studios'].tolist() for studio in sublist]
        studio_frequency = Counter(all_studios)

        studios = [studio for studio, count in studio_frequency.most_common()]
        counts = [count for studio, count in studio_frequency.most_common()]

        plt.figure(figsize=(10, 6))
        sns.barplot(x=counts, y=studios, color='skyblue')
        plt.title("Favorite Animation Studio")
        plt.xlabel("Frequency")
        plt.ylabel("Studio")
        plt.show()


    def compare_scores(self, df: pd.DataFrame):
        average_user_score = np.mean(df['user_score'])
        average_public_score = np.mean(df['public_score'])

        plt.figure(figsize=(8, 4))
        scores = ['User Score', 'Public Score']
        avg_scores = [average_user_score, average_public_score]
        colors = ['skyblue', 'lightgreen']
        plt.bar(scores, avg_scores, color=colors)
        plt.title('Comparison of User Score and Public Score')
        plt.xlabel('Score Type')
        plt.ylabel('Average Score')
        plt.ylim(0, 10)
        plt.show()

    def compare_score_distribution(self, df: pd.DataFrame):
        plt.figure(figsize=(10, 6))
        sns.kdeplot(data=df, x='user_score', label='User Score', fill=True)
        sns.kdeplot(data=df, x='public_score', label='Public Score', fill=True)
        
        plt.title('Distribution of User Score vs. Public Score')
        plt.xlabel('Score')
        plt.ylabel('Density')
        plt.legend()
        plt.show()

    def watching_frequency_by_day(self, df: pd.DataFrame):
        df['day_of_week'] = df['user_created_at'].dt.day_name()
        days_frequency = df['day_of_week'].value_counts()

        plt.figure(figsize=(10, 6))
        sns.countplot(x='day_of_week', data=df, order=days_frequency.index, palette='viridis')
        plt.title('Anime Watching Frequency by Day of Week')
        plt.xlabel('Day of Week')
        plt.ylabel('Number of Animes Watched')
        plt.show()


def load_data(user_id: str, disable_proxy: bool) -> pd.DataFrame:
    try:
        df = pd.read_pickle(f"{user_id}_anime_data.pkl")
        anime_fetcher = AnimeDataFetcher(user_id, disable_proxy)
        df = anime_fetcher.fix_data_types(df)
    except FileNotFoundError:
        anime_fetcher = AnimeDataFetcher(user_id, disable_proxy)
        df = anime_fetcher.download_anime_data()
    return df


def main(user_id: str, disable_proxy: bool):
    df = load_data(user_id, disable_proxy)
    
    visualizer = AnimeDataVisualizer()
    visualizer.visualize_data(df)

    analyzer = AnimeDataAnalyzer()

    analyzer.genre_preferences_over_time(df)
    analyzer.genre_preferences_over_time_dt(df)
    analyzer.favorite_studio(df)
    analyzer.compare_scores(df)
    analyzer.compare_score_distribution(df)
    analyzer.watching_frequency_by_day(df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('user_id', help='User ID for which to download and visualize anime data.')
    parser.add_argument('--disable-proxy', action='store_true', help='Disable the use of proxies for requests.')
    args = parser.parse_args()
    main(args.user_id, args.disable_proxy)
