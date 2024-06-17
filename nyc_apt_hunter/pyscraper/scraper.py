from bs4 import BeautifulSoup
import requests
import time
import numpy as np


def make_request(
    url: str,
    min_delay: int = 0.5,
    max_delay: int = 3.5,
    timeout: int = 15,
    retry_attempt: bool = False,
):
    """Web scraper for a given URL with request rate control built in

        Args:
            url (str): URL to request
            min_delay (int, optional): Minimum time to sleep for request rate 
                                    control.
            max_delay (int, optional): Max time to sleep for request rate control
            timeout (int, optional): Max timeout setting for request
            retry_attempt (bool, optional): True/False variable to check if this is a retry after a timeout error

        Returns:
            BeautifulSoup: Raw HTML from request module. None is returned if status code is not 200.
            res.status_code: Status code from request to check if request was successful.
        """

    user_agents = [
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_5) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/13.1.1 Safari/605.1.15",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:77.0) Gecko/20100101 Firefox/77.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:77.0) Gecko/20100101 Firefox/77.0",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36",
    ]

    # Control the request rate
    time.sleep(round(np.random.uniform(min_delay, max_delay)))

    # Pick a random user agent and set headers
    user_agent = np.random.choice(user_agents)
    headers = {"User-Agent": user_agent}

    # Try/except block to execute the request and handle any exceptions
    try:
        res = requests.get(url, allow_redirects=False, timeout=timeout)
        # headers=headers, timeout=timeout)
        # If the response was successful, no Exception will be raised
        res.raise_for_status()
    # except requests.exceptions.Timeout:
    #     print("Timeout error occurred")

    #     # Try request again after a delay
    #     # If retry attempt is true, need to exit method
    #     if retry_attempt == False:
    #         time.sleep(round(np.random.uniform(10, 20)))
    #         make_request(url, retry_attempt=True)
    #     else:
    #         soup = None
    #         status_code = 404
    #         return soup, status_code
    except Exception as e:
        print(f"An error occured, {e}, while trying to access the URL: {url}")
        soup = None
        status_code = 404
        return soup, status_code

    # Verify that the request was successful
    if res.status_code == 200:
        soup = BeautifulSoup(res.content, "lxml")
    elif res.status_code == 403:
        raise (f"A status code of {res.status_code} occured.")
    else:
        soup = None

    return soup, res.status_code


def generate_page_URL(
        BASE_URL: str,
        min_price: int,
        max_price: int,
        page_num: int
):
    """Generates a page URL with the current price range and page number"""
    return BASE_URL + f"{min_price}-to-{max_price}/{page_num}/"
