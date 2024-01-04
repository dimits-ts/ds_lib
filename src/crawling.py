from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium import webdriver
from urllib.request import Request, urlopen
import bs4
import threading


def fetch_soup(url: str) -> bs4.BeautifulSoup:
    """
    Fetch and parse the HTML content of a web page using BeautifulSoup.

    :param url: The URL of the web page to be fetched and parsed.
    :type url: str
    :return: A BeautifulSoup object representing the parsed HTML content.
    :rtype: bs4.BeautifulSoup
    """
    req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    html_page = urlopen(req).read()
    return bs4.BeautifulSoup(html_page, 'html.parser')


class ChromeDriverManager:
    """
    A singleton class for managing a Chrome WebDriver instance using Selenium.

    This class ensures that only one instance of the Chrome WebDriver is created
    and provides methods for setting options, getting the instance, and quitting
    the WebDriver when done.

    Usage:
    - To set options, use the `set_options` class method.
    - To obtain the Chrome WebDriver instance, use the `get` class method.
    - To quit and close the Chrome WebDriver instance, use the `quit` class method.
    """
    DRIVER = None
    lock = threading.Lock()
    options = webdriver.ChromeOptions()

    @classmethod
    def set_options(cls, options):
        cls.options = options
        
    @classmethod   
    def get(cls):
        """
        Get the singleton instance of the Chrome WebDriver.

        If the WebDriver instance does not exist, a new one is created using the
        configured options.

        Returns:
        The singleton instance of the Chrome WebDriver.
        """
        with cls.lock:
            if cls.DRIVER is None:
                print("Creating new driver...")
                cls.DRIVER = _new_chrome_driver(cls.options)
                print("New driver online.")
            return cls.DRIVER

    @classmethod  
    def quit(cls):
        """
        Quit and close the Chrome WebDriver instance.

        This method should be called when the WebDriver is no longer needed.
        """
        with cls.lock:
            if cls.DRIVER is not None:
                cls.DRIVER.quit()
                cls.DRIVER = None

    def __init__(self):
        """
        Raise an exception to enforce the singleton pattern.

        To obtain the Chrome WebDriver instance, use the `get` class method.
        """
        raise Exception("DriverManager is a singleton. Use DriverManager.get() to get a driver.")


def default_options():
    """
    Default configuration, featuring headless setup, standard language and user agent settings.
    """
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument('--lang=en')
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/104.0.0.0 Safari/537.36")
    return options


def jupyter_options():
    """
    Adds jupyter-specific settings to default_options().
    """
    options = default_options()
    # prevent console spam
    options.add_experimental_option('excludeSwitches', ['enable-logging'])
    return options


def _new_chrome_driver(options):
    try:
        service = Service()
    except Exception:
        service = Service(ChromeDriverManager().install())
        
    return webdriver.Chrome(service=service, options=options)