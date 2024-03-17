from requests import get
from bs4 import BeautifulSoup
from collections import deque
from concurrent.futures import ThreadPoolExecutor, wait
from threading import Lock
import json


# my included packages
import re


class IMDbCrawler:
    """
    put your own user agent in the headers
    """
    headers = {
        'User-Agent': 'IMDB Crawler'
    }
    top_250_URL = 'https://www.imdb.com/chart/top/'
    main_URL = 'https://www.imdb.com'

    def __init__(self, crawling_threshold=1000):
        """
        Initialize the crawler

        Parameters
        ----------
        crawling_threshold: int
            The number of pages to crawl
        """
        # TODO
        self.crawling_threshold = crawling_threshold
        self.not_crawled = deque()
        self.crawled = deque()
        self.added_ids = set()
        self.not_crawled_lock = Lock()
        self.crawled_lock = Lock()

    def get_id_from_URL(URL):
        """
        Get the id from the URL of the site. The id is what comes exactly after title.
        for example the id for the movie https://www.imdb.com/title/tt0111161/?ref_=chttp_t_1 is tt0111161.

        Parameters
        ----------
        URL: str
            The URL of the site
        Returns
        ----------
        str
            The id of the site
        """
        id_pattern = r'\/title\/(\w+)'
        matched = re.search(id_pattern, URL)
        if matched is None: # couldnt find id
            return None
        return matched.group(1)

    def write_to_file_as_json(self):
        """
        Save the crawled files into json
        """
        with open('../IMDB_crawled.json', 'w') as f:
            json.dump(list(self.crawled), f)

        with open('../IMDB_not_crawled.json', 'w') as f:
            json.dump(list(self.not_crawled), f)

        with open('../IMDB_added_ids.json', 'w') as f:
            json.dump(list(self.added_ids), f)

    def read_from_file_as_json(self):
        """
        Read the crawled files from json
        """
        with open('../IMDB_crawled.json', 'r') as f:
            self.crawled = deque(json.load(f))

        with open('../IMDB_not_crawled.json', 'r') as f:
            self.not_crawled = deque(json.load(f))

        with open('../IMDB_added_ids.json', 'r') as f:
            self.added_ids = set(json.load(f))

    def crawl(self, URL):
        """
        Make a get request to the URL and return the response

        Parameters
        ----------
        URL: str
            The URL of the site
        Returns
        ----------
        requests.models.Response
            The response of the get request
        """
        response = get(URL, headers=IMDbCrawler.headers)
        return response
    
    def add_ids_to_queue(self, ids):
        for id in ids:
            if id is None:
                continue
            try:
                self.not_crawled_lock.acquire()
                if id in self.added_ids:
                    continue
                self.not_crawled.append(f'{IMDbCrawler.main_URL}/title/{id}')
                self.added_ids.add(id)
            finally:
                self.not_crawled_lock.release()


    def extract_top_250(self):
        """
        Extract the top 250 movies from the top 250 page and use them as seed for the crawler to start crawling.
        """
        response = self.crawl(IMDbCrawler.top_250_URL)
        if response.status_code != 200:
            raise Exception(f'Response code {response.status_code}, expected 200.')
        soup = BeautifulSoup(response.content, 'html.parser')
        ids = IMDbCrawler.get_links(soup)
        self.add_ids_to_queue(ids)


    def get_imdb_instance(self):
        return {
            'id': None,  # str
            'title': None,  # str
            'first_page_summary': None,  # str
            'release_year': None,  # str
            'mpaa': None,  # str
            'budget': None,  # str
            'gross_worldwide': None,  # str
            'rating': None,  # str
            'directors': None,  # List[str]
            'writers': None,  # List[str]
            'stars': None,  # List[str]
            'related_links': None,  # List[str]
            'genres': None,  # List[str]
            'languages': None,  # List[str]
            'countries_of_origin': None,  # List[str]
            'summaries': None,  # List[str]
            'synopsis': None,  # List[str]
            'reviews': None,  # List[List[str]]
        }

    def start_crawling(self):
        """
        Start crawling the movies until the crawling threshold is reached.
        TODO: 
            replace WHILE_LOOP_CONSTRAINTS with the proper constraints for the while loop.
            replace NEW_URL with the new URL to crawl.
            replace THERE_IS_NOTHING_TO_CRAWL with the condition to check if there is nothing to crawl.
            delete help variables.

        ThreadPoolExecutor is used to make the crawler faster by using multiple threads to crawl the pages.
        You are free to use it or not. If used, not to forget safe access to the shared resources.
        """

        self.extract_top_250()
        futures = []
        crawled_counter = 0

        with ThreadPoolExecutor(max_workers=20) as executor:
            while crawled_counter < self.crawling_threshold:
                self.not_crawled_lock.acquire()
                try:
                    URL = self.not_crawled.pop()
                except IndexError:
                    continue
                except Exception as e:
                    print(f'Exception in main Thread: {e}')
                finally:
                    self.not_crawled_lock.release()
                futures.append(executor.submit(self.crawl_page_info, URL))
                crawled_counter += 1
                # print(crawled_counter)
                # if crawled_counter % 20 == 0:
                #     wait(futures)
                #     futures = []
                #     print(crawled_counter)
            wait(futures)

    def crawl_page_info(self, URL):
        """
        Main Logic of the crawler. It crawls the page and extracts the information of the movie.
        Use related links of a movie to crawl more movies.
        
        Parameters
        ----------
        URL: str
            The URL of the site
        """
        response = self.crawl(URL)
        retries = 0
        while response.status_code == 504 and retries < 20:
            response = self.crawl(URL)
            retries += 1
        if response.status_code != 200:
            print(f'returend status code {response.status_code} for URL {URL}')
            return
        movie = self.get_imdb_instance()
        try:
            self.crawled_lock.acquire()
            self.crawled.append(movie)
        finally:
            self.crawled_lock.release()
        self.extract_movie_info(response, movie, URL)
        self.add_new_movies_to_queue(response)

    
    def add_new_movies_to_queue(self, res):
        soup = BeautifulSoup(res.content, 'html.parser')
        ids = IMDbCrawler.get_links(soup)
        self.add_ids_to_queue(ids)
        

    def extract_movie_info(self, res, movie, URL):
        """
        Extract the information of the movie from the response and save it in the movie instance.

        Parameters
        ----------
        res: requests.models.Response
            The response of the get request
        movie: dict
            The instance of the movie
        URL: str
            The URL of the site
        """
        
        soup = BeautifulSoup(res.content, 'html.parser')

        movie['id'] = IMDbCrawler.get_id_from_URL(URL)
        movie['title'] = IMDbCrawler.get_title(soup)
        movie['first_page_summary'] = IMDbCrawler.get_first_page_summary(soup)
        movie['release_year'] = IMDbCrawler.get_release_year(soup)
        movie['mpaa'] = IMDbCrawler.get_mpaa(URL)
        movie['budget'] = IMDbCrawler.get_budget(soup)
        movie['gross_worldwide'] = IMDbCrawler.get_gross_worldwide(soup)
        movie['directors'] = IMDbCrawler.get_director(soup)
        movie['writers'] = IMDbCrawler.get_writers(soup)
        movie['stars'] = IMDbCrawler.get_stars(soup)
        movie['related_links'] = IMDbCrawler.get_related_links(soup)
        movie['genres'] = IMDbCrawler.get_genres(soup)
        movie['languages'] = IMDbCrawler.get_languages(soup)
        movie['countries_of_origin'] = IMDbCrawler.get_countries_of_origin(soup)
        movie['rating'] =  IMDbCrawler.get_rating(soup)
        movie['summaries'] = IMDbCrawler.get_summary(URL)
        movie['synopsis'] = IMDbCrawler.get_synopsis(URL)
        movie['reviews'] = IMDbCrawler.get_reviews_with_scores(URL)

    def get_links(soup):
        ids = list(map(lambda x: IMDbCrawler.get_id_from_URL(x.get("href")), soup.find_all("a")))
        return ids

    def get_summary_link(url):
        """
        Get the link to the summary page of the movie
        Example:
        https://www.imdb.com/title/tt0111161/ is the page
        https://www.imdb.com/title/tt0111161/plotsummary is the summary page

        Parameters
        ----------
        url: str
            The URL of the site
        Returns
        ----------
        str
            The URL of the summary page
        """
        try:
            return f'{url}/plotsummary'
        except:
            print("failed to get summary link")

    def get_review_link(url):
        """
        Get the link to the review page of the movie
        Example:
        https://www.imdb.com/title/tt0111161/ is the page
        https://www.imdb.com/title/tt0111161/reviews is the review page
        """
        try:
            return f'{url}/reviews'
        except:
            print("failed to get review link")

    def get_title(soup):
        """
        Get the title of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The title of the movie

        """
        try:
            return soup.title.text
        except:
            print("failed to get title")

    def get_first_page_summary(soup):
        """
        Get the first page summary of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The first page summary of the movie
        """
        try:
            return soup.find('p', {'data-testid': 'plot'}).text
        except:
            print("failed to get first page summary")

    def get_director(soup):
        """
        Get the directors of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The directors of the movie
        """
        try:
            credits = soup.find('div', {'class':"sc-67fa2588-3 fZhuJ"}).find('ul').find_all('li', {'data-testid': 'title-pc-principal-credit'})
            for i in range(len(credits)):
                if 'Director' in credits[i].text:
                    return [j.text for j in credits[i].find_all('li')]
            raise Exception('Couldnt find director')
        except:
            print("failed to get director")

    def get_stars(soup):
        """
        Get the stars of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The stars of the movie
        """
        try:
            credits = soup.find('div', {'class':"sc-67fa2588-3 fZhuJ"}).find('ul').find_all('li', {'data-testid': 'title-pc-principal-credit'})
            for i in range(len(credits)):
                if 'Stars' in credits[i].text:
                    return [j.text for j in credits[i].find_all('li')]
            raise Exception('Couldnt find Stars')
        except:
            print("failed to get stars")

    def get_writers(soup):
        """
        Get the writers of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The writers of the movie
        """
        try:
            credits = soup.find('div', {'class':"sc-67fa2588-3"}).find('ul').find_all('li', {'data-testid': 'title-pc-principal-credit'})
            for i in range(len(credits)):
                if 'Writer' in credits[i].text:
                    return [j.text for j in credits[i].find_all('li')]
            raise Exception('Couldnt find writers')
        except:
            print("failed to get writers")

    def get_related_links(soup):
        """
        Get the related links of the movie from the More like this section of the page from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The related links of the movie
        """
        try:
            return list(map(lambda x: x.get('href'), soup.find('section', {'data-testid':"MoreLikeThis"}).find_all("a")))
        except:
            print("failed to get related links")

    def get_summary(URL):
        """
        Get the summary of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The summary of the movie
        """
        try:
            response = IMDbCrawler.crawl(None, IMDbCrawler.get_summary_link(URL))
            if response.status_code != 200:
                raise Exception(f'encountered status code {response.status_code}, expected 200.')
            soup = BeautifulSoup(response.content, 'html.parser')
            return [i.text for i in soup.find('div', {'data-testid':"sub-section-summaries"}).find_all('li')]
        except:
            print("failed to get summary")

    def get_synopsis(URL):
        """
        Get the synopsis of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The synopsis of the movie
        """
        try:
            response = IMDbCrawler.crawl(None, IMDbCrawler.get_summary_link(URL))
            if response.status_code != 200:
                raise Exception(f'encountered status code {response.status_code}, expected 200.')
            soup = BeautifulSoup(response.content, 'html.parser')
            return [i.text for i in soup.find('div', {'data-testid':"sub-section-synopsis"}).find_all('li')]
        except:
            print("failed to get synopsis")

    def get_review_and_score(soup):
        try:
            review = soup.find('div', {'class':'content'}).text
            score = soup.find('span', {'class':'rating-other-user-rating'}).find('span').text
        except:
            return None
        return [review, score]

    def get_reviews_with_scores(URL):
        """
        Get the reviews of the movie from the soup
        reviews structure: [[review,score]]

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[List[str]]
            The reviews of the movie
        """
        try:
            response = IMDbCrawler.crawl(None, IMDbCrawler.get_review_link(URL))
            if response.status_code != 200:
                raise Exception(f'encountered status code {response.status_code}, expected 200.')
            soup = BeautifulSoup(response.content, 'html.parser')
            reviews = soup.find_all('div', {'class':"imdb-user-review"})
            return list(filter(None, [IMDbCrawler.get_review_and_score(review) for review in reviews]))
        except:
            print("failed to get reviews")

    def get_genres(soup):
        """
        Get the genres of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The genres of the movie
        """
        try:
            return [i.text for i in soup.find('div', {'data-testid':"genres"}).find_all('a')]
        except:
            print("Failed to get generes")

    def get_rating(soup):
        """
        Get the rating of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The rating of the movie
        """
        try:
            return soup.find('div', {'data-testid':"hero-rating-bar__aggregate-rating__score"}).text.split('/')[0]
        except:
            print("failed to get rating")

    def get_mpaa(URL):
        """
        Get the MPAA of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The MPAA of the movie
        """
        try:
            response = IMDbCrawler.crawl(None, f'{URL}/parentalguide')
            soup = BeautifulSoup(response.content, 'html.parser')
            return soup.find('tr', {'id':"mpaa-rating"}).find_all('td')[1].text
        except:
            print("failed to get mpaa")

    def get_release_year(soup):
        """
        Get the release year of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The release year of the movie
        """
        try:
            return soup.find('a', href=lambda x: x and '/releaseinfo' in x).text
        except:
            print("failed to get release year")

    def get_languages(soup):
        """
        Get the languages of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The languages of the movie
        """
        try:
            return [i.text for i in soup.find('li', {'data-testid':"title-details-languages"}).find_all('li')]
        except:
            print("failed to get languages")
            return None

    def get_countries_of_origin(soup):
        """
        Get the countries of origin of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The countries of origin of the movie
        """
        try:
            return [i.text for i in soup.find('li', {'data-testid':"title-details-origin"}).find_all('li')]
        except:
            print("failed to get countries of origin")

    def get_budget(soup):
        """
        Get the budget of the movie from box office section of the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The budget of the movie
        """
        try:
            boxoffice = soup.find('div', {'data-testid':"title-boxoffice-section"}).text
            budget_pattern = r'Budget\$([\d,]+)'
            matched = re.search(budget_pattern, boxoffice)
            return matched.group(1)
        except:
            print("failed to get budget")

    def get_gross_worldwide(soup):
        """
        Get the gross worldwide of the movie from box office section of the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The gross worldwide of the movie
        """
        try:
            boxoffice = soup.find('div', {'data-testid':"title-boxoffice-section"}).text
            budget_pattern = r'Gross worldwide\$([\d,]+)'
            matched = re.search(budget_pattern, boxoffice)
            return matched.group(1)
        except:
            print("failed to get gross worldwide")


def main():
    imdb_crawler = IMDbCrawler(crawling_threshold=600)
    # imdb_crawler.read_from_file_as_json()
    imdb_crawler.start_crawling()
    imdb_crawler.write_to_file_as_json()


if __name__ == '__main__':
    main()
