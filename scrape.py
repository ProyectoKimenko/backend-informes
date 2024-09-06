# from selenium import webdriver
# from selenium.webdriver.common.by import By
# from selenium.webdriver.common.desired_capabilities import DesiredCapabilities

# def fetch_page_title(url):
#     # Set up the Selenium WebDriver to connect to the running selenium container
#     options = webdriver.ChromeOptions()
#     options.add_argument("--no-sandbox")
#     options.add_argument("--headless")  # Run in headless mode since we don't need a UI
#     options.add_argument("--disable-dev-shm-usage")

#     # Connect to the Selenium container
#     driver = webdriver.Remote(
#         command_executor='http://selenium:4444/wd/hub',  # URL of Selenium in docker-compose
#         options=options
#     )

#     # Open the webpage
#     driver.get(url)

#     # Get and print the page title
#     page_title = driver.title
#     print("Page title is:", page_title)

#     # Close the browser session
#     driver.quit()
    
#     return page_title

from selenium import webdriver
from selenium.webdriver.common.by import By

def fetch_page_title(url: str):
    options = webdriver.ChromeOptions()
    options.add_argument("--no-sandbox")
    # options.add_argument("--headless")
    options.add_argument("--disable-dev-shm-usage")

    driver = webdriver.Remote(
        command_executor='http://selenium:4444/wd/hub',
        options=options
    )

    try:
        driver.get(url)

        cookies = [
            {
                "domain": ".flowreporter.com",
                "expirationDate": 1757163381,
                "httpOnly": False,
                "name": "_clck",
                "path": "/",
                "secure": False,
                "value": "t5ftre%7C2%7Cfoy%7C0%7C1698"
            },
            {
                "domain": ".flowreporter.com",
                "expirationDate": 1725714409,
                "httpOnly": False,
                "name": "_clsk",
                "path": "/",
                "secure": False,
                "value": "11zmybq%7C1725628009985%7C2%7C1%7Cu.clarity.ms%2Fcollect"
            },
            {
                "domain": "flowreporter.com",
                "httpOnly": True,
                "name": "auth0_session_0",
                "path": "/",
                "sameSite": "lax",
                "secure": False,
                "value": "%7B%22tag%22%3A%220ozcG7jOJjzQUzSDT%2B4OOQ%3D%3D%22%7D"
            },
            {
                "domain": "flowreporter.com",
                "httpOnly": True,
                "name": "auth0_session_1",
                "path": "/",
                "sameSite": "lax",
                "secure": False,
                "value": "C%2FwD9lbzNJ2Nr4j7IgwgMjtoJBKdy43Kyeitvu3a9AMmhNYM7f8oVAVfebKUb6j"
            },
            {
                "domain": "flowreporter.com",
                "expirationDate": 1727210404.677734,
                "httpOnly": False,
                "name": "CSRFToken",
                "path": "/",
                "secure": False,
                "value": "d9a8d2dae9632c3db838a3de1503680f54918f471fb8c333a626d081be6ae0c2"
            },
            {
                "domain": "flowreporter.com",
                "httpOnly": False,
                "name": "PHPSESSID",
                "path": "/",
                "secure": False,
                "value": "et0m24n82veqn92eunds81vb3r"
            }
        ]

        # Add each cookie to the browser session
        for cookie in cookies:
            # Remove "expirationDate" as it's not valid for `add_cookie()`
            cookie.pop("expirationDate", None)
            driver.add_cookie(cookie)

        # Reload the page after adding cookies
        driver.get(url)

        page_title = driver.title
    except Exception as e:
        print(f"An error occurred: {e}")
        page_title = None
    finally:
        driver.quit()
    
    return page_title
