# MISO Policy Tracker Project

This document summarizes the technical work and progress for the MISO Hackathon project. Our goal is to use public data from multiple sources to find early, corroborated signals of emerging energy policy themes relevant to MISO's operating territory.

---

### ## How to Run This Project (For Teammates)

Follow these steps to set up and run the project on your own machine.

#### **Prerequisites:**
* You must have **Python 3** installed on your computer.
* You must have **Visual Studio Code** installed.

#### **Step 1: Set Up the Project Folder**
1.  Download and unzip the project folder.
2.  Open the **`miso-policy-tracker`** folder in Visual Studio Code.

#### **Step 2: Configure the Environment**
1.  Open a new terminal in VS Code (**Terminal** > **New Terminal**).
2.  Create a virtual environment by running this command:
    ```bash
    python3 -m venv venv
    ```
3.  Activate the environment:
    ```bash
    source venv/bin/activate
    ```
4.  Install all the necessary Python libraries from the `requirements.txt` file:
    ```bash
    pip3 install -r requirements.txt
    ```

#### **Step 3: Add API Keys**
You will need to get free developer API keys for all three data sources and add them to the corresponding scripts.

1.  **Reddit API:** Follow the steps in "Part 1" below to get your keys and add them to `02_reddit_scraper.ipynb`.
2.  **X (Twitter) API:** Follow the steps in "Part 1" to get your **Bearer Token** and add it to `05_twitter_scraper.ipynb`.
3.  **NewsAPI.org:** Get a free API key from [NewsAPI.org](https://newsapi.org/) and add it to `08_news_scraper.ipynb`.

#### **Step 4: Run the Final Pipeline**
To get the final result, you only need to run the data collection scripts and then the final analysis script.

1.  **Run the Scrapers:** Run these three notebooks to collect the data. They will create the `.csv` files in the `/data/raw` folder.
    * `02_reddit_scraper.ipynb`
    * `05_twitter_scraper.ipynb`
    * `08_news_scraper.ipynb`
2.  **Run the Final Analysis:** Once the data is collected, run this notebook to produce the final, corroborated signal and chart.
    * `10_final_signal_analysis.ipynb` (or `09_final_signal_analysis.ipynb` if you are using that one)

*Note: The other notebooks (`01`, `03`, `04`, etc.) were part of the development process and show our work. They do not need to be run to get the final result.*

---

### ## Project Documentation: The Workflow

Our project pipeline is broken into three main parts:

### **Part 1: Setting Up API Credentials**
To gather data, we set up developer accounts and obtained API keys for three distinct platforms:
* **Reddit:** To capture public discussion and sentiment.
* **X (Twitter):** To capture real-time news and commentary.
* **NewsAPI.org:** To capture articles from official news media.

### **Part 2: Data Collection (The Scrapers)**
We created a separate, geo-filtered scraper for each data source.

* **`02_reddit_scraper.ipynb`:** Scrapes the `/r/energy` subreddit and then filters the posts to keep only those that mention states or provinces within MISO's territory.
* **`05_twitter_scraper.ipynb`:** Uses the X API to search for tweets that contain both energy-related keywords and the names of MISO states.
* **`08_news_scraper.ipynb`:** Uses the NewsAPI to search for news articles that contain both energy-related keywords and the names of MISO states.

### **Part 3: Final Analysis and Visualization**
This is the final step where we find the signal.

* **`10_final_signal_analysis.ipynb`:** This notebook is the heart of the project. It loads the data from all three sources, identifies the "corroborated keywords" that appear in both social media (Reddit/X) and official news, and generates the final chart visualizing this powerful, cross-platform signal.