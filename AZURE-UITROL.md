# Veilige Azure-uitrol

Deze handleiding bereidt de chatbot voor op gebruik via een browser. Er is nog niets naar Azure gestuurd en er zijn geen cloudresources gemaakt.

## Voorgestelde opzet

- **Azure Container Apps (Consumption)** draait de Streamlit-app. Stel minimaal 0 en maximaal 1 replica in. De app schaalt dan naar nul wanneer ze niet gebruikt wordt; de eerste opening kan daardoor even duren.
- **Azure OpenAI** maakt embeddings voor semantisch zoeken. Optioneel gebruikt de modus “Kort antwoord” daarnaast een chatmodeldeployment om een beknopt, bronvermeld antwoord uit gevonden passages te formuleren.
- **GitHub Container Registry (GHCR)** bewaart de publieke containerimage met app, Codex-PDF en vooraf gebouwde Chroma-index. Er zijn geen API-sleutels in de image. Azure Container Apps kan deze publieke image ophalen zonder ACR.
- **Toegangscode** schermt de app af. Zet de toegangscode alleen als `APP_ACCESS_PASSWORD`-secret in Container Apps. Schakel HTTPS afgedwongen in en deel de code alleen privé met de gebruiker. Azure Container Apps biedt daarnaast ingebouwde aanmelding; gebruik die als extra beveiligingslaag en beperk toegang waar mogelijk tot de bedoelde gebruiker.

De containerimage is openbaar downloadbaar. Dat is hier aanvaardbaar omdat de app-code en PDF publiek mogen zijn; `.env` en lokale Chroma-databases gaan niet mee. Alleen `deployment/chroma_seed` wordt bewust opgenomen. De website zelf blijft beschermd met Azure-aanmelding en de app-toegangscode. De maximumschaal van 1 replica houdt het computegebruik voor één vriend eenvoudig en voorspelbaar.

## Kosten eerst controleren

1. Maak zelf een Azure-account aan via [Azure Free Account](https://azure.microsoft.com/free/). Microsoft vraagt bij registratie om een telefoonnummer en betaalkaart of niet-prepaid debetkaart voor identiteitscontrole. Maak geen Pay-as-you-go-upgrade als je daar nog geen toestemming voor geeft.
2. De gratis accountvoorwaarden en actuele limieten veranderen. Controleer die in Azure Portal voordat je resources maakt. Na de proefperiode of het tegoed worden de diensten uitgeschakeld tenzij je zelf overstapt op Pay-as-you-go. De maandelijkse gratis hoeveelheid van Container Apps geldt niet automatisch voor Azure OpenAI of Log Analytics.
3. Azure-budgets sturen waarschuwingen; ze stoppen uitgaven niet automatisch. Een gratis tegoed is ook geen harde uitgavenlimiet als je later naar betalen naar gebruik overstapt.
4. Gebruik in Cost Management een budget met meldingen bij bijvoorbeeld 50%, 80% en 100%. Controleer eventuele kosten voor loggegevens en AI-gebruik.
5. Voor Azure OpenAI is modelgebruik doorgaans per token geprijsd. Stel waar mogelijk quota en lage snelheidslimieten in, beperk toegang tot de app en controleer de prijs voor de gekozen regio en modellen.

GitHub vermeldt publieke packages en containerimage-opslag/-bandbreedte momenteel als gratis. [Controleer de actuele GitHub Packages-regels](https://docs.github.com/en/billing/concepts/product-billing/github-packages), want voorwaarden kunnen veranderen. Een gratis Azure-account of tegoed garandeert niet dat Azure-gebruik gratis blijft. Bekijk de [Azure-prijscalculator](https://azure.microsoft.com/pricing/calculator/), [Container Apps-prijzen](https://learn.microsoft.com/en-us/azure/container-apps/billing) en [Azure OpenAI-prijzen](https://azure.microsoft.com/pricing/details/azure-openai/) voor de gekozen regio voordat je de app publiek maakt.

## Eenmalig instellen

### 1. Account en regio

Maak het Azure-account zelf aan en meld je aan in Azure Portal. Kies later één regio waarin zowel Container Apps als de gewenste Azure OpenAI-modellen beschikbaar zijn. Modelnamen, quota en beschikbaarheid verschillen per regio en kunnen veranderen.

### 2. Azure OpenAI

Maak in Azure AI Foundry een Azure OpenAI-resource en één embeddingdeployment voor vectorzoekopdrachten, bijvoorbeeld een beschikbaar klein embeddingmodel. Kies voor **Standard / betalen per gebruik**; kies geen Provisioned Throughput Units, omdat die capaciteit ook zonder vragen een vaste kost kan hebben. Voor de modus “Kort antwoord” maak je daarnaast een chatmodeldeployment, bijvoorbeeld GPT-5 mini. Dit kost extra tokens per antwoord; gebruik deze modus alleen wanneer Camil een direct antwoord nodig heeft.

Noteer de **deploymentnaam** (die kan verschillen van de modelnaam), het endpoint en de API-sleutel. De app gebruikt het Azure OpenAI v1-endpoint; daarvoor is geen gedateerde API-versie nodig. Zet deze waarden niet in screenshots, Git of berichten. Voor het eenmalig bouwen van de embeddings moet de sleutel tijdelijk beschikbaar zijn op jouw eigen computer: zet de Azure-instellingen alleen in het lokale `.env`-bestand. Dat bestand wordt niet in Git of de containerimage opgenomen. Gebruik `Prepare-Azure-Index.bat` om de index te maken. De sleutel wordt alleen gebruikt om embeddings aan te vragen en staat niet in de gemaakte index.

### 3. Containerimage en app

Maak de containerimage met de Dockerfile in deze projectmap en publiceer haar als **publieke package** op GitHub Container Registry (`ghcr.io`). Azure Container Apps ondersteunt publieke GHCR-images en kan die zonder registry-wachtwoord ophalen. Publiceer geen `.env`, sleutels of andere vertrouwelijke bestanden.

Maak daarna een Consumption-omgeving en een Container App die de publieke `ghcr.io/...`-image gebruikt. Stel de HTTP-ingress in op poort `8501` met HTTPS; laat onbeveiligd HTTP-verkeer uitgeschakeld. Stel de schaal in op minimum `0` en maximum `1` replica. Gebruik een publieke image als app-invoer; de webtoegang blijft via aanmelding en toegangscode beveiligd.

Stel de volgende gewone omgevingsvariabelen in:

```text
AI_PROVIDER=azure
REQUIRE_ACCESS_PASSWORD=true
AZURE_OPENAI_ENDPOINT=<endpoint van je resource>
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=<embeddingdeploymentnaam>
AZURE_OPENAI_CHAT_DEPLOYMENT=<chatdeploymentnaam>
```

Voeg de sleutels als **secrets** toe in Container Apps. Geef de secret zelf een kleine naam, bijvoorbeeld `azure-openai-api-key` en `app-access-password`, en koppel ze daarna aan deze omgevingsvariabelen:

```text
AZURE_OPENAI_API_KEY
APP_ACCESS_PASSWORD
```

Kopieer de daadwerkelijke waarden niet in de projectbestanden. In de Container Apps-instellingen kies je bij de betreffende omgevingsvariabele voor de secret-verwijzing.

Genereer voor de toegangscode een lange willekeurige waarde (minstens 32 willekeurige tekens). Sla die alleen op in de secret store en deel hem privé met je vriend. De app blijft ontoegankelijk zolang de toegangscode ontbreekt.

Schakel in Container Apps ook de ingebouwde aanmelding in en vereis aanmelden voordat gebruikers de app bereiken. Configureer zo mogelijk dat alleen de account van je vriend toegang krijgt. De app-toegangscode blijft dan een extra laag. Azure beschrijft de ingebouwde aanmelding [hier](https://learn.microsoft.com/en-us/azure/container-apps/authentication).

### 4. Chroma-index in de cloudimage

Stel bij Container Apps ook deze omgevingsvariabelen in:

```text
CHROMA_DB_ROOT=/app/deployment/chroma_seed
REQUIRE_PREBUILT_INDEX=true
```

De index moet vóór de Docker-image worden voorbereid. `Prepare-Azure-Index.bat` gebruikt Docker Desktop en de gegevens uit `.env`; de eenmalige embeddings worden aangerekend volgens Azure OpenAI-gebruik. Als de index ontbreekt, stopt cloudvectorzoeken met een duidelijke fout in plaats van stilzwijgend embeddings te maken en mogelijk opnieuw kosten te veroorzaken. Bij vervanging van de PDF, wijziging van de artikelparser of wijziging van de embeddingdeployment maak je een nieuwe index en bouw je een nieuwe image.

### 5. Controle vóór delen

- Open de app in een privévenster: zonder toegangscode mag geen document of zoekvraag bereikbaar zijn.
- Log in met de gedeelde code en controleer één artikelnummer en één themavraag.
- Kijk na of de verwachte Chroma-records in de app zichtbaar zijn.
- Controleer Cost Management en modelquota na de eerste opbouw van de index en na gebruik van “Kort antwoord”; die modus doet naast embeddings ook chatmodelaanvragen.
- Deel pas daarna de HTTPS-link met je vriend.

## Voorwaarden voordat de uitrol kan beginnen

Er is een Azure-account nodig. Dit account moet door de eigenaar zelf worden gemaakt en de eigenaar moet de regio, eventuele betaalinstellingen en budgetmeldingen nakijken. Zodra dat klaar is, kunnen we de instellingen stap voor stap invullen en eerst de verwachte terugkerende kosten bekijken. Start de resources pas na die controle.
