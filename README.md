# Chatbot voor de Codex over het welzijn op het werk

Een eenvoudige Nederlandstalige webapp om artikelnummers en thema's in de meegeleverde Codex-PDF op te zoeken.

## Starten met Docker Desktop

1. Installeer en open Docker Desktop.
2. Dubbelklik op `Start-chatbot.bat`. De eerste start bouwt de app en opent die daarna in de browser.
3. Artikelnummers worden rechtstreeks opgezocht. Thematische vragen gebruiken semantisch vectorzoeken; stel daarvoor AI_PROVIDER=azure in .env in met je Azure Foundry-instellingen.
4. Gebruik `Stop-chatbot.bat` om de chatbot af te sluiten.

Laat Docker Desktop geopend zolang je de app gebruikt.

## Wat de app kan

- Een artikelnummer zoals `I.2-6` rechtstreeks opzoeken en de volledige tekst tonen.
- Een inhoudelijke vraag semantisch vergelijken met artikelteksten, ook wanneer de vraagwoorden niet letterlijk in het artikel staan.
- Artikelen semantisch doorzoeken op genummerde onderdelen; treffers worden samengevoegd per artikel, de volledige tekst wordt getoond en alleen het best passende onderdeel wordt gemarkeerd als er een opsplitsing is.
- Alle artikelen met minstens één onderdeel boven de instelbare drempel tonen, met overeenkomstscore. Er wordt geen extra AI-antwoord gegenereerd.
- Zoekbare artikelonderdelen opslaan in een persistente Chroma-vector database.
- Optioneel een toegangscode vragen voordat documenten of AI-functies worden geladen.

De vector database wordt bij de eerste themavraag met de gekozen embeddings-API opgebouwd en lokaal bewaard. Dit kan de eerste keer even duren; latere vragen hergebruiken de index. Stel REQUIRE_PREBUILT_INDEX=true in voor een cloudcontainer die alleen een vooraf gebouwde index mag gebruiken. Een nieuwe PDF of een ander embeddingmodel maakt automatisch een aparte index aan. Er is geen letterlijke woordmatchlijst: zoekresultaten worden op semantische overeenkomst gerangschikt. De overeenkomstscore is geen accuracy van het antwoord. Stel de drempel af met representatieve vragen: lager geeft meer resultaten, hoger filtert sterker. Bekijk bij belangrijke vragen daarom ook de volledige wettekst.

## API-sleutel

Bewaar API-sleutels alleen in het lokale .env-bestand. Deel dat bestand niet en zet het niet in Git of in de Dockerfile. Exacte artikelnummers opzoeken werkt zonder sleutel. Semantisch vectorzoeken gebruikt alleen de gekozen embeddings-API en kan kosten veroorzaken; er is geen chatmodeldeployment nodig. In Azure horen API-sleutels uitsluitend in de secret-instellingen van Container Apps, nooit in de broncode of de containerimage.

## Cloudgebruik

Volg `AZURE-UITROL.md` om Azure OpenAI en Azure Container Apps in te stellen. Er is nog niets uitgerold. De handleiding gebruikt een publieke image op GitHub Container Registry; de app-code en Codex zijn daarin openbaar, maar de API-sleutels blijven als Azure-secrets buiten de image.

De Azure-index wordt vooraf gemaakt en in de containerimage opgenomen. Gebruik daarvoor `Prepare-Azure-Index.bat` nadat je Azure OpenAI-gegevens in het lokale `.env`-bestand hebt ingevuld. De app van je vriend gebruikt daarna alleen de browser; de reken- en modelaanvragen lopen via Azure.

## De PDF vervangen

Vervang `data/Codex_over_het_welzijn_op_het_werk.pdf` door de nieuwe PDF en bouw de container opnieuw met `docker compose up --build`.

## Twee manieren om vragen te stellen

De standaardmodus toont de relevante volledige Codex-artikelen en gebruikt de threshold-schuif. Zet “Kort antwoord geven in plaats van de artikellijst” aan om de volledige vectorindex te doorzoeken en de volledige tekst van de beste kandidaatartikelen door het chatmodel te laten beoordelen. De threshold-schuif verdwijnt in die modus. Het antwoord noemt de bronartikelen en hoort geen antwoord te verzinnen als de artikelteksten onvoldoende bewijs bevatten.

Voor korte antwoorden is naast de embeddingdeployment ook de omgevingsvariabele AZURE_OPENAI_CHAT_DEPLOYMENT nodig (bijvoorbeeld gpt-5-mini). Elke korte beantwoording gebruikt het chatmodel en kan dus Azure-kosten veroorzaken. Juist/Fout is optioneel: bij Fout kan Camil een verduidelijkingsvraag beantwoorden, waarna de app opnieuw in de Codex zoekt. Deze feedback blijft alleen in de huidige sessie en traint het model niet automatisch.