import unirest
response = unirest.get("https://webknox-question-answering.p.rapidapi.com/questions/answers?answerLookup=false&answerSearch=false&question=Where+is+Los+Angeles%3F",
  headers={
    "X-RapidAPI-Key": "ifwPBHiHmSmshWZC6pwPDaScEWzip1Qwzthjsnd7ZpODSJRa0L"
  }
)
print response.body