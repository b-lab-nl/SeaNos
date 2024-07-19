# loads config with the settings for the lyrics generation
#
from src.lyrics import Lyrics
from src.news import NewsAPI, RSSParser
from src.speak import Speak
from src.utils import load_config, load_env
from src.summarise import Summary

import os
import re
import sys
import argparse
import logging
from time import sleep
from datetime import datetime
import random
import sqlite3

from typing import Literal, Union

# initialise the logger
logging.basicConfig(level=logging.INFO)
fh = logging.FileHandler("lyrics.log")
fh.setLevel(logging.INFO)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger = logging.getLogger(__name__)
logger.addHandler(fh)
logger.setLevel(logging.INFO)


def close_handlers():
    for handler in logger.handlers:
        handler.close()
        logger.removeHandler(handler)
#####################################################


# TODO: make sqlite db class to feed the news/lyrics


def get_news(query: str = None,
             newsSelection: Union[Literal['all', 'random'], int] = 'random',
             newsSource: str = 'RSS',
             topNews: bool = False,
             configpath: str = 'config.yaml'):
    logger.info("Starting the TTS process")
    logger.info(datetime.now())

    keys = load_env()
    if newsSource == 'newsapi':
        NewsGetter = NewsAPI(api_key=keys['NEWS_API_KEY'])
        if topNews == True:
            newsList = NewsGetter.get_top_news(query)
        else:
            newsList = NewsGetter.get_any_news(query)
    elif newsSource == 'RSS':
        RSSGetter = RSSParser(config_path=configpath)
        newsList = RSSGetter.parse_feeds()

    if len(newsList) == 0:
        logger.error(f"No news found for q:{query}")
        newsString = "Please be kind to us."

    if newsSelection == 'random':
        d = random.choice(newsList)
        newsString = f'Title:{d["title"]},Description:{d["description"]}'
    elif newsSelection == 'all':
        newsString = " ".join(
            [f'Title:{d["title"]},Description:{d["description"]}'
             for d in newsList])
    elif newsSelection.isdigit():
        newsString = " ".join(
            [f'Title:{d["title"]},Description:{d["description"]}'
             for d in newsList[:int(newsSelection)]])
    else:
        raise ValueError("newsSelection must be 'random', 'all' or an integer")

    return newsString


def get_summary(text: str = None, debug: bool = False) -> str:
    logger.info("Summarising the news")
    logger.info(datetime.now())

    srizer = Summary()
    srizer.initialise_openai()
    summary_text, OAI_summary = srizer.Summarise(text)

    if debug:
        logger.info(f"OAI_summary:{OAI_summary}")
    return summary_text


def get_phonetics(text: str = None):
    logger.info("Extracting the phonetics of the lyrics")
    logger.info(datetime.now())

    phonetics = Phonetics(lyrics_text, source_lang="en", target_lang="gle")
    lyrics_text_converted = phonetics.convert_text(lyrics_text)
    return lyrics_text_converted


def get_speech(text: str = None,
               language: str = "en",
               configpath: str = "config.yaml",
               modelpath: str = None,
               outpath: str = None):
    logger.info("Starting the TTS process")
    logger.info(datetime.now())

    assert (text is not None), "Text must be provided for TTS"
    assert (outpath is not None), "Output path must be provided for TTS"

    speaker = Speak(language=language,
                    config_path=configpath,
                    model_path=modelpath)
    speaker.speak(lyrics_text, OutPath=outpath)
    return True


def get_lyrics(text: str = None,
               query: str = None,
               debug: bool = False,
               configpath: str = "config.yaml"):
    logger.info("Starting the lyrics generation process")
    logger.info(datetime.now())

    lyricist = Lyrics(config=load_config(config_path=configpath))
    lyricist.initialise_openai()
    print("Query:", query)
    if query is not None:
        lyricist.news_topic = query

    lyrics, OAI_lyrics = lyricist.generate(text)
    if debug:
        logger.info(f"OAI_lyrics:{OAI_lyrics}")

    # remove quotes
    lyrics = lyrics.replace('"', '')
    lyrics = lyrics.replace("'", "")

    return lyrics


def write_sql(query: str = None,
              newsSource: str = None,
              newsSelection: str = None,
              language: str = None,
              rawNews: str = None,
              summaryOfNews: str = None,
              lyrics_text: str = None,
              outpath: str = None):
    """
    Write the parameters to an sqlite db
    """
    logger.info("Accessing the database")
    logger.info(datetime.now())

    conn = sqlite3.connect('artifacts/lyrics.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS lyrics
                 (creation_date text, query text, newsSource text, newsSelection text, language text, raw text, summary text, lyrics text, outpath text)''')
    conn.commit()

    insert_statement = f'''INSERT INTO lyrics VALUES ("{str(datetime.now())}","{query}","{newsSource}","{newsSelection}","{language}","{rawNews}","{summaryOfNews}","{lyrics_text}","{outpath}")'''
    try:
        c.execute(insert_statement)
    except Exception as e:
        logger.error(
            f"Error in writing lyrics to db:{e}, insert statement:{insert_statement}")
    conn.commit()
    return True


def fetch_latest(query: str = None,
                 column: Literal['raw', 'summary', 'lyrics']=None) -> str:
    '''
    Fetch latest raw text from the SQLite database with this query, or random selection if query has zero matches

    :param query: str
    :return: str
    '''

    logger.info("Accessing the database -- for fetching raw news texts")
    logger.info(datetime.now())

    conn = sqlite3.connect('artifacts/lyrics.db')
    cursor = conn.cursor()

    if query:
        cursor.execute(f"SELECT {column} FROM lyrics WHERE query = ? ORDER BY creation_data DESC LIMIT 1", (query,))
        result = cursor.fetchone()

        if result:
            txt = result[0]
        else:
            cursor.execute(f"SELECT {column} FROM lyrics ORDER BY RANDOM() LIMIT 1")
            result = cursor.fetchone()
            txt = result[0] if result else None
    else:
        cursor.execute(f"SELECT {column} FROM lyrics ORDER BY RANDOM() LIMIT 1")
        result = cursor.fetchone()
        txt = result[0] if result else None

    conn.close()

    return txt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", default="Headline news",
                        help="Query to generate lyrics for")
    parser.add_argument("--debug", type=bool, help="Debug mode", default=False)
    parser.add_argument("--refresh", type=int, help="Refresh rate in seconds",
                        default=None)
    parser.add_argument("--topNews", type=bool,
                        help="Get top news", default=False)
    parser.add_argument("--newsSource", type=str,
                        help="News source", default='RSS')
    parser.add_argument("--newsSelection", type=str,
                        help="News selection", default='random')
    parser.add_argument("--language", type=str, default="gle")
    parser.add_argument("--summary", type=bool,
                        help="Summarise the news", default=False)

    config_path = "config.yaml"

    args = parser.parse_args()
    query = args.query
    debug = args.debug
    refresh_rate = args.refresh
    topNews = args.topNews
    newsSource = args.newsSource
    newsSelection = args.newsSelection
    summary = args.summary
    language = args.language

    CONFIG = load_config(config_path=config_path)
    FALLBACK_LYRICS = CONFIG['fallback_lyrics']

    while True:
        # STEP 1
        sleep(20)
        try:
            newsTexts = get_news(query=query,
                                 newsSelection=newsSelection,
                                 newsSource=newsSource,
                                 topNews=topNews,
                                 configpath=config_path)
        except Exception as e:
            logger.error(e)
            # TODO: if fail fetch latest from SQL with same query, or random.
            newsTexts = fetch_latest(query=query, column='raw')
            if newsTexts is None:
                logger.error(f"Query: {query} gave zero results from the lyrics DB and the News fetcher failed")
                newsTexts = '''The world is burning. No nature, no future.'''

        # STEP 2
        try:
            summaryOfNews = get_summary(newsTexts, debug=debug)
        except Exception as e:
            logger.error(e)
            # TODO: if fail fetch from SQL
            summaryOfNews = fetch_latest(query=query, column='summary')
            if summaryOfNews is None:
                logger.error(f"Query: {query} gave zero results from the lyrics DB and the Summary fetcher failed")
                summaryOfNews = '''The world is burning. No nature, no future.'''

        # STEP 3
        try:
            lyrics_text = get_lyrics(text=summaryOfNews,
                                     query=query,
                                     debug=debug,
                                     configpath=config_path)
        except Exception as e:
            logger.error(e)
            # TODO: if fail fetch from SQL
            lyrics_text = fetch_latest(query=query, column='lyrics')
            if summaryOfNews is None:
                logger.error(f"Query: {query} gave zero results from the lyrics DB and the Lyrics fetcher failed")
                lyrics_text = FALLBACK_LYRICS

        # text to phonetics
        # phonetics = Phonetics(lyrics_text, source_lang="en", target_lang="gle")
        # lyrics_text_converted = phonetics.convert_text(lyrics_text)

        # outpath is the output path for the TTS, should be randomly generated with datetime
        outpath = f"artifacts/{language}_{query}_{newsSource}_{datetime.now().strftime('%Y%m%d%H%M%S')}.wav"
        outpath = outpath.replace(" ", "_")
        # We want to store the output path in an sqlite db,
        # together with the lyrics, the news text and the parameters

        # language: gle (Irish)
        # STEP 4
        try:
            speak = get_speech(text=lyrics_text,
                               language=language,
                               modelpath=None,
                               configpath=config_path,
                               outpath=outpath)
        except Exception as e:
            logger.error(e)
            outpath="ERROR"

        # WRITE TO DB
        write_sql(query=query,
                  newsSource=newsSource,
                  newsSelection=newsSelection,
                  language=language,
                  rawNews=newsTexts,
                  summaryOfNews=summaryOfNews,
                  lyrics_text=lyrics_text,
                  outpath=outpath)


