# Fact Check Agent

I built this for part 2 of the CogCulture product management assignment. You upload a PDF, it picks out the numbers, dates and other hard claims, looks each one up on the web, and tells you which ones are right, which are out of date and which look made up.

Live app: (paste your Streamlit link here)

Every claim ends up with one of three labels:

- Verified: the web agrees with it
- Inaccurate: it was true at some point, or it is close, but the number is off. The app shows the current figure and a corrected version of the sentence
- False: nothing credible backs it up

There is also an "Unchecked" label for when a lookup fails (rate limit, bad reply). Those are left out of the score so they don't count against the document.

## How it works

It's a three step pipeline.

1. pdfplumber reads the PDF page by page, so each claim can say which page it came from.
2. A language model (Llama 3.3 on Groq) reads the text and lists the claims worth checking: stats, dates, money, technical specs. It also writes a search query for each one.
3. Each query goes to Tavily, which searches the live web. The model then reads the results, picks a label and writes the correct fact. The source links on each card are the pages Tavily returned, so the app never shows a link the model made up.

Both services have a free tier, and the app talks to them with plain HTTP requests (the `requests` library), so there are no extra SDKs to install.

The two API keys are entered in the sidebar. They are never written into the code. If they are stored in the hosting platform's secrets instead, the boxes disappear and visitors can test the app without entering anything.

Other things in the app: a trust score for the whole document, filters by status and category, a search box, a strictness setting (how picky it is about rounding), a paste-text option, a built-in sample document for quick demos, and downloads as CSV, JSON or a markdown report.

## Running it on your own computer

You need Python 3.10 or newer, a free key from https://console.groq.com and a free key from https://tavily.com

```
pip install -r requirements.txt
```

You can paste the keys into the sidebar when the app opens. To avoid pasting them every time, make a folder called `.streamlit` next to `app.py` and put a file called `secrets.toml` in it:

```
GROQ_API_KEY = "your-groq-key"
TAVILY_API_KEY = "your-tavily-key"
```

Then start it:

```
streamlit run app.py
```

`secrets.toml` is in `.gitignore`, so it won't be pushed to GitHub. Keep it that way.

## Deploying on Streamlit Cloud

1. Push the repo to GitHub (without secrets.toml).
2. On share.streamlit.io choose New app, pick the repo and set the main file to `app.py`.
3. Under Advanced settings, Secrets, paste the same two lines as above.
4. Deploy and wait a couple of minutes.

If you ever want to use a different model provider, add `LLM_BASE_URL`, `LLM_MODEL` and `LLM_API_KEY` to the secrets. Anything that follows the OpenAI chat format will work.

## Things that don't work well yet

- Scanned PDFs. There's no OCR, so image-only files come back empty.
- Free tiers have rate and daily limits. A long document takes a while, and if you see lots of "Unchecked" results, wait a minute and use fewer claims.
- Only the first 24,000 characters of a document are read, and files over 10 MB are rejected.
- It's only as good as what the search finds. For small or obscure companies it sometimes finds nothing, and then the claim gets marked False when it's really just unknown. Check the source links.

## What I'd add next

- OCR for scanned PDFs
- Check a claim against the date of the document, so "latest figure" claims are judged by the right year
- A fix-it mode that exports the document with the corrected sentences
- Let people paste a URL instead of a PDF

