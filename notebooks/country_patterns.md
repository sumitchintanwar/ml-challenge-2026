# Deep Dive: Country-Specific Noise & Transformation Patterns

## Executive Summary

This empirical analysis investigates the exact transformation and noise patterns between reference entities (`Source 1`) and their true matched counterparts in `Source 2` and `Source 3` as defined by [train_ground_truth.tsv](file:///home/sumitchint_work/amazon-ml-challenge/data/raw/train_ground_truth.tsv).

We sampled **30 ground-truth matched pairs from the United States** and **30 ground-truth matched pairs from India**, analyzing name and address mutations side-by-side. The findings systematically expose failure modes of naive string matching and directly prescribe our feature engineering architecture.

---

## 1. Summary of Discovered Transformation Patterns

### 1.1 United States Specific Patterns
1. **Address Block Inversion (State/City First)**: In S1, addresses standardly follow `[Number] [Street], [City], [State]`. In S2/S3, addresses frequently invert to `[State], [City], [Street]` (e.g., `ME, Madison, 476 Horsetail Hill Road`) or `[City], [Street], [State]` (`ELGIN, 1287-B FLEETWOOD DR, IL`).
2. **Street Number Modifications**: True matches frequently introduce street number ranges or letter suffixes: `674 Ridge Gate Drive` $\to$ `674-678 RIDGE GATE DR` or `1287 Fleetwood Drive` $\to$ `1287-B FLEETWOOD DR`.
3. **Unit / Suite Normalization Discrepancies**: S1 records contain explicit secondary clauses like `Unit 308`, which are often condensed, appended to street numbers (`1287-B`), or completely omitted in S2/S3.
4. **Web Domains & DBAs in Business Name**: S2/S3 names frequently contain appended URLs or DBA strings (`Vargas Phoenix` $\to$ `Vargas Phoenix | www.vargaspho.com`).
5. **Township & Municipal Descriptors**: City names vary between municipality types: `Ardmore` $\to$ `ARDMORE TOWNSHIP`.
6. **Phonetic & Typographical OCR Typos**: High-consonant typographical shifts (`Zaon Irenic` $\to$ `Zaon Icr`).

### 1.2 India Specific Patterns
1. **Non-Latin Script Transliteration**: Matches in S2/S3 frequently feature native regional scripts (Tamil, Devanagari Hindi, Kannada, Telugu, Punjabi) paired with English reference entities (e.g., `Sree Construction Private Limited` $\to$ `ஸ்ரீ கன்ஸ்ட்ரக்ஷன் பிரைவேட் லிமிடெட்`).
2. **Displaced & Permuted Legal Designations**: Severe word reordering within legal affixes (e.g., `Sree Construction Private Limited` $\to$ `Sree Limited Private Construction` or `Private Beyond Heat (Limited)`).
3. **Landmark Inclusion, Omission & Variations**: Landmarks (e.g., `Near Vishal Hall`, `Near Parimal Railway Crossing`, `Opp. RTA Office`) appear in one source and are omitted or altered in another.
4. **Door / House / Plot / Khasra Numbering Variations**: Numbering format shifts from `137` to `A-137` or `Flat No. 403` to `403` or house number typos (`45, Teli Gulli` $\to$ `44, TELI GULLI`).
5. **State / UT Naming Multiplicity**: Alternation between full English state name, 2-letter state code, and vernacular name (`Maharashtra` $\to$ `MH` $\to$ `महाराष्ट्र`).
6. **Phonetic Spelling Substitutions**: Common phoneme variations: `Sree` $\leftrightarrow$ `Shree`, `Laxmi` $\leftrightarrow$ `Lakshmi`, `Kishan` $\leftrightarrow$ `Krishan`.

---

## 2. United States: 30 Matched Pairs (Side-by-Side)

| # | S1 ID & Matched ID | S1 Business Name | Matched Business Name | S1 Address | Matched Address | Identified Transformations |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | `S1-618905093`<br>$\to$ `S2-325149773` (Source 2) | Vava Center | **Ariazephfaye** | 1466 Macedonia Road, Ardmore, AL | **1466 MACEDONIA RD, ARDMORE TOWNSHIP, AL** | Lexical variation / typo / alias |
| 2 | `S1-121625780`<br>$\to$ `S2-452719994` (Source 2) | Zaon Irenic LLC | **Zaon Írenic Llc** | 1287 Fleetwood Drive, Unit 308, Elgin, IL | **ELGIN, 1287-B FLEETWOOD DR, IL** | Non-Latin/Indic script or accent in name; Lexical variation / typo / alias; Street/Door number variation (range or unit) |
| 3 | `S1-425013944`<br>$\to$ `S2-386487296` (Source 2) | Vargas Phoenix LLC | **Vargas Phoenix \| www.vargaspho.com** | 674 Ridge Gate Drive, Brownsburg, IN | **674-678 RIDGE GATE DR, BROWNSBURG, IN** | URL / Domain appended to name; Lexical variation / typo / alias; Street/Door number variation (range or unit) |
| 4 | `S1-98912952`<br>$\to$ `S2-407672921` (Source 2) | Lilias Thomas, O.D. | **Lilias Thomas, O.D. (Ltd) #75638** | ME, Madison, 476 Horsetail Hill Road | **<NULL>, 00476 HORSETAIL HILL ROAD, ME, MAISON** | Legal suffix/prefix addition or omission; Street/Door number variation (range or unit) |
| 5 | `S1-683271240`<br>$\to$ `S2-691282936` (Source 2) | Womens Health Physicians LLC | **WOMENS-HEALTH LLC PHYSICIANS** | 2831 Hancock Street, Lake Station, IN | **2831 HANCOCK ST, LAKE STATION, IN** | Lexical variation / typo / alias |
| 6 | `S1-80181441`<br>$\to$ `S2-212725369` (Source 2) | Nexify LLC | **NEXIFY LLC LLC** | 46 Derby Street, Valley Stream, NY | ***[NULL]*** | Name word transposition/reordering; Target address MISSING (Null) |
| 7 | `S1-749337518`<br>$\to$ `S2-867793801` (Source 2) | Kennedy Endocrinology LLC | **kennedyendocrinology.com** | 6 Shadow Mountain Road, NM, El Prado | **#6 SHADOW MOUNTAIN RD, EL PRADO, NM** | URL / Domain appended to name; Lexical variation / typo / alias; Street/Door number variation (range or unit) |
| 8 | `S1-293613057`<br>$\to$ `S2-907550602` (Source 2) | Lowry Apex Disciplined Inc | **Inc Lowry Apex Dbisciplined** | 14 Fisher Street, Fl Floor 1, Buffalo, NY | **#14 FISHER SAINT, BUFFALO, NY** | Lexical variation / typo / alias; Street/Door number variation (range or unit) |
| 9 | `S1-233286158`<br>$\to$ `S2-266066109` (Source 2) | A Cure 3 IT | **A CURE 3 [IT]** | 7404 Adele Drive, Norfolk City, VA | **NORFOLLK, 7404 ADELE DRIVE, VA** | Lexical variation / typo / alias |
| 10 | `S1-252401816`<br>$\to$ `S2-353192911` (Source 2) | Torey's Allied Development | **TOREY'S ALLIED DEVEBDEMNT** | 300 Main Street, Marion, IL | **30 MAIN STREET, MARION, IL** | Lexical variation / typo / alias; Street/Door number variation (range or unit) |
| 11 | `S1-5304135`<br>$\to$ `S2-797083171` (Source 2) | Brown Teucrium Inc. | **Brown Teucrium Ínc.** | 619 Oak Ridge Lane, Mars Hill, NC | **619 OAK RIDGE LANE, MARS HILL, NC** | Non-Latin/Indic script or accent in name; Lexical variation / typo / alias; Address case variation |
| 12 | `S1-223482820`<br>$\to$ `S2-886667267` (Source 2) | EP Chile | **EP Chtel** | 1681 Great Hill Road, Guilford, CT | **1681 Great Hill Road, GUILFORD, CT** | Lexical variation / typo / alias; Address case variation |
| 13 | `S1-461075361`<br>$\to$ `S2-967073708` (Source 2) | Zenith Pgim LLC | **LLC Zenith Pgím** | Goodyear, AZ, 14278 Harvard Street | **#14278 HARVARD ST, GOODYEAR, AZ** | Non-Latin/Indic script or accent in name; Lexical variation / typo / alias; Street/Door number variation (range or unit) |
| 14 | `S1-892236211`<br>$\to$ `S2-978107596` (Source 2) | First Presbyterian Church of Minneapolis | **firstpresbyterianchurch.com** | 3912 Bloomington Avenue, Minneapolis, MN | **3912 BLOOMINGTON AVE, MINNEAPOLIS, MN** | URL / Domain appended to name; Lexical variation / typo / alias |
| 15 | `S1-386236216`<br>$\to$ `S2-765091111` (Source 2) | Luna Empire Viii LLC | **luna empire viii llc** | 400 Old Franklin Turnpike, Unit 107, Rocky Mount Town, VA | **400 OLD FRANKLIN TPKE, VA, ROCKY MOUNT TOWN** | Case variation only; Street/Door number variation (range or unit) |
| 16 | `S1-448253755`<br>$\to$ `S2-117403445` (Source 2) | Classic Apex Ridge, LLC | **CLASSIC APEX RIDGE,** | 2900 Hamilton Church Road, Unit 201, Nashville, TN | **2900 HAMILTON CHURCH RD, NASHVILLE, TN** | Legal suffix/prefix addition or omission; Street/Door number variation (range or unit) |
| 17 | `S1-417801692`<br>$\to$ `S3-591739955` (Source 3) | Bernardina Holland, DPM | **Bernardina Holland,** | IL, Sauk Village, 2203 A 220th Street | **2203 A 220th Street, Chicago Heights CDP, Illinois** | Legal suffix/prefix addition or omission |
| 18 | `S1-569997956`<br>$\to$ `S2-117571831` (Source 2) | Tri-State Reliable Plum Inc | **Tri-State Reliable Psltum  Inc** | NY, Southold, 1625 Hortons Lane | **1625 HORTONS LANE, SOUTHOLD, NY** | Lexical variation / typo / alias; Address component inversion/reordering |
| 19 | `S1-54868449`<br>$\to$ `S2-387453591` (Source 2) | Jones School of Medicine LLC | **Jones School of Medicine  LLC** | IL, 314 Fulton Street, Lacon | **LACON, IL, FULTON ST** | Name word transposition/reordering; Street/Door number variation (range or unit) |
| 20 | `S1-599594570`<br>$\to$ `S2-696449515` (Source 2) | A Cure 3 You | **A 3 Cure You** | 420 28th Street, Ankeny, IA | **420  28ST ST, PMB 2317, ANKENY, IA** | Name word transposition/reordering; Street/Door number variation (range or unit) |
| 21 | `S1-873257940`<br>$\to$ `S2-94953018` (Source 2) | Link Colonial Concepts | **Link Concepts Colonial** | 7330 192nd Place, Lynnwood, WA | **7330. 192ND PLACE, PMB 7006, LYNNWOOD, WA** | Name word transposition/reordering; Street/Door number variation (range or unit) |
| 22 | `S1-904003142`<br>$\to$ `S2-88415347` (Source 2) | Leach Prairie Pharmaceuticals LLC | **Leach Prairie  Pharmaceuticals LLC** | 3001 Quaker Creek Drive, NC, Mebane | **MEBANE, NC, QUAKER CREEK DR** | Name word transposition/reordering; Street/Door number variation (range or unit) |
| 23 | `S1-492010115`<br>$\to$ `S2-827956590` (Source 2) | Tawil Games Inc | **TAWILGAMES.COM** | 101 Wagon Lane, Angleton, TX | **101 WAGON LANE, ANGLETON, TX** | URL / Domain appended to name; Lexical variation / typo / alias; Address case variation |
| 24 | `S1-834551788`<br>$\to$ `S2-226101602` (Source 2) | Metro Guild | **Metro Guild** | 469 Jamessie Lane, Russellville, KY | ***[NULL]*** | Exact name match; Target address MISSING (Null) |
| 25 | `S1-476659981`<br>$\to$ `S3-161230697` (Source 3) | Dermatology Partners Inc. | **dermatologypartners.com** | 3105 Fairway Woods, Sanford, NC | **Sanford, North Carolina, 3105 Fairway Woods** | URL / Domain appended to name; Lexical variation / typo / alias |
| 26 | `S1-225534770`<br>$\to$ `S2-221221448` (Source 2) | National Canada L.L.C. | **National Canada** | 39 Orchard Road, Dalton, MA | **39 ORCHABD RD, DALTON, MA** | Legal suffix/prefix addition or omission |
| 27 | `S1-654135003`<br>$\to$ `S2-31001588` (Source 2) | Fernanda's Advertising LLC | **Fernanda's Advertising** | 6543 Sexton Drive, VA, Chesterfield County | **6543 SEXTON DR, PMB 8714, CHESTEERFIELD CITY, VA** | Legal suffix/prefix addition or omission; Street/Door number variation (range or unit) |
| 28 | `S1-292914005`<br>$\to$ `S2-679886766` (Source 2) | Springdale Cornerstone Pharmaceuticals | **Springdale Pharmaceuticals Cornerstone** | 573 Box Canyon Road, Springdale, UT | **573 BOX CANYON RD, SPRINGDALE, UT** | Name word transposition/reordering |
| 29 | `S1-447481150`<br>$\to$ `S2-580477255` (Source 2) | Bandon Vision Grand Associates | **BANDON VISION ASSOCIATES GRAND** | 88336 Pacific Surf Lane, Bandon, OR | **OR, 88336 PACIFIC SURF LANE, BANDON** | Name word transposition/reordering; Address component inversion/reordering |
| 30 | `S1-451577354`<br>$\to$ `S3-786499920` (Source 3) | Ember Holding | **Umbravio** | 905 Cedartree Lane, Unit APT 5, Claymont, DE | **Claymont, Delaware, 905 Cedartree Ln, # APT 5** | Lexical variation / typo / alias; Street/Door number variation (range or unit) |

---

## 3. India: 30 Matched Pairs (Side-by-Side)

| # | S1 ID & Matched ID | S1 Business Name | Matched Business Name | S1 Address | Matched Address | Identified Transformations |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | `S1-760331148`<br>$\to$ `S2-233443928` (Source 2) | Usha Group | **USHA GROUP** | 45, Teli Gulli Park Rd, Andheri East, Near Vishal Hall, Mumbai, Maharashtra | **44, TELI GULLI PARK RD, ANDHERI EAST, NEAR VISHAL HALL, MUMBAI, Maharashtra** | Case variation only; Landmark addition/omission; Street/Door number variation (range or unit) |
| 2 | `S1-658639312`<br>$\to$ `S2-855493631` (Source 2) | Sree Construction Private Limited | **ஸ்ரீ கன்ஸ்ட்ரக்ஷன் பிரைவேட் லிமிடெட்** | 11/5 Voc Street Alagappan Nagar, Madurai, Tamil Nadu | **11/5 VOC STREET ALAGAPPAN NAGAR, MADURAI, Tamil Nadu** | Non-Latin/Indic script or accent in name; Lexical variation / typo / alias; Address case variation |
| 3 | `S1-407310445`<br>$\to$ `S2-266534441` (Source 2) | Beyond Heat Private Limited | **Private Beyond Heat (Limited)** | 137, First Floor, Vishawkarma Nagar-2, Maha, Jaipur, Rajasthan | **A-137, FIRST FLOOR, VISHAWKARMA NAGAR-2, MAHA, Rajasthan, JAIPUR** | Lexical variation / typo / alias; Street/Door number variation (range or unit) |
| 4 | `S1-825023193`<br>$\to$ `S2-131131789` (Source 2) | Future Constructions Private Limited | **फ्यूचर कंस्ट्रक्शंस प्राइवेट लिमिटेड** | Fl-J-104, Kewale, Saomya, Fortune Infra Ventures, Panvel, Raigarh(Mh), Maharashtra | **PANVEL, FL-J-104., Maharashtra, RAIGARH(MH), KEWALE, SAOMYA, FORTUNE INFRA VENTURES** | Non-Latin/Indic script or accent in name; Lexical variation / typo / alias; Street/Door number variation (range or unit) |
| 5 | `S1-666734313`<br>$\to$ `S2-183359415` (Source 2) | Aditya Foods Private Limited | **आदित्य फूड्स प्राइवेट लिमिटेड** | G-5B, Ganga Path, Durga Marg, Bani Park, Jaipur, Rajasthan | **G-5B, JAIPUR, Rajasthan** | Non-Latin/Indic script or accent in name; Lexical variation / typo / alias |
| 6 | `S1-390996499`<br>$\to$ `S2-953753647` (Source 2) | Devan Healthcare Pvt Ltd | **devanhealthcare.com** | Telangana, Tirumalagiri, H:No 29-712, Vinayaka Nagar, Neredmet, Hyderabad | **H:NO 29-712, VINAYAKA NAGAR, NEREDMET, TIRUMALAGIRI, Andhra Pradesh** | URL / Domain appended to name; Lexical variation / typo / alias |
| 7 | `S1-953760777`<br>$\to$ `S2-325128410` (Source 2) | Arihant Logistics Private Limited | **অরিহন্ত লজিস্টিকস প্রাইভেট লিমিটেড** | Bl -1, 13Th Floor, Fl-13D 76/1B Bidhan Sarani, Kolkata, Kolkata, Howrah, West Bengal | **13TH FLOOR, FL-13D 76/1B BIDHAN SARANI, KOLKATA, HOWRAH, KOLKATA, West Bengal, BL -1** | Non-Latin/Indic script or accent in name; Lexical variation / typo / alias; Street/Door number variation (range or unit); Address component inversion/reordering |
| 8 | `S1-595488395`<br>$\to$ `S2-673553081` (Source 2) | Black Infotech Pvt Ltd | **బ్లాక్ ఇన్ఫోటెక్ ప్రైవేట్ లిమిటెడ్** | 10-58, Vinayaka Nagar Bala Nagar, Hyderabad, Telangana | **10-58, VINAYAKA NAGAR BALA NAGAR, HYDERABAD, Telangana** | Non-Latin/Indic script or accent in name; Lexical variation / typo / alias; Address case variation |
| 9 | `S1-190247294`<br>$\to$ `S2-340822435` (Source 2) | Krishna Products Private Limited | **कृष्ण प्रोडक्ट्स प्राइवेट लिमिटेड** | New Delhi, 19, New Delhi, Arunachal Building, 914-915, 9Th Floor, Barakhamba Road, C P, Delhi | **19, NEW DELHI, दिल्ली** | Non-Latin/Indic script or accent in name; Lexical variation / typo / alias; Street/Door number variation (range or unit) |
| 10 | `S1-133980545`<br>$\to$ `S3-434818565` (Source 3) | Southern Media Pvt Ltd | **Southern Media** | 15, Alkapuri Amba Mata, Udaipur, Rajasthan | **#15, Alkapuri Amba Mata, Udaipur, राजस्थान** | Legal suffix/prefix addition or omission; Street/Door number variation (range or unit) |
| 11 | `S1-521491126`<br>$\to$ `S2-95480559` (Source 2) | Engineering Al Plastics Private Limited | **ENGINEERING AL PLASTICS PRIVATE** | 10, Clive Road 1St Floor, Kolkata, Howrah, West Bengal | **10, CLIVE ROAD 1ST FLOOR, KOLKATA, পশ্চিমবঙ্গ** | Legal suffix/prefix addition or omission; Street/Door number variation (range or unit) |
| 12 | `S1-603986604`<br>$\to$ `S2-848358987` (Source 2) | Mumbai Inland Pvt Ltd | **Mumbai Ltd Pvt Inland** | D-3004, Floor-30, Lodha Primero, Nm Joshi Marg, Apollo Mills Compd Mahalaxmi, Jacob Circ, Le, Mumbai, Mumbai City, Maharashtra | ***[NULL]*** | Name word transposition/reordering; Target address MISSING (Null) |
| 13 | `S1-298907561`<br>$\to$ `S2-845788207` (Source 2) | Sky International Private Limited | **સ્કાય ઇન્ટરનેશનલ પ્રાઇવેટ લિમિટેડ** | 36, Parthbhumi Society, Nr. Govt. Tubeweel, Bopal, Ahmedabad, Gujarat | **36, AHMEDABAD HQ REGION, AHMEDABAD, Gujarat** | Non-Latin/Indic script or accent in name; Lexical variation / typo / alias |
| 14 | `S1-935638022`<br>$\to$ `S2-405574665` (Source 2) | Gimpex Minerals Private Limited | **GIMPEX PRIVATE-LIMITED SERVICES** | 11 Meena Bagh Dayal Bagh, Agra, Uttar Pradesh | **11 MEENA BAGH DAYAL BAGH, AGRA, Uttar Pradesh** | Lexical variation / typo / alias; Address case variation |
| 15 | `S1-942967942`<br>$\to$ `S3-322685656` (Source 3) | Vsn Naturals (India) | **vsnnaturalsindia.com** | Jayalakshmi Estates, 5Th Floor, 8 Haddows Road, Madras, Chennai, Tamil Nadu | **Jayalakshmi Estates, 5Th Floor, 8 Haddows Road, Chennai, Madras, தமிழ்நாடு** | URL / Domain appended to name; Lexical variation / typo / alias |
| 16 | `S1-978647130`<br>$\to$ `S2-973351735` (Source 2) | Gwalior Urology Pvt Ltd | **-- GWALIOR ULRLYG PVT LTD** | 181, Gudi Payaga, Balajipuram, Guda Gudi Ka Naka Lashkar, Gwalior, Madhya Pradesh | **181, GWALIOR, Madhya Pradesh** | Lexical variation / typo / alias |
| 17 | `S1-71032511`<br>$\to$ `S2-327298544` (Source 2) | Yellow Collective Limited | **Yellow Collective** | Top Floor, Ward No 3, 705/D - 22A, G. R. Apartmets Mehrauli, New Delhi, South Delhi, Delhi | **TOP FLOOR, WARD NO 3, 705/D - 22A, G. R. APARTMETS MEHRAULI, NEW DELHI, दिल्ली** | Legal suffix/prefix addition or omission |
| 18 | `S1-696867509`<br>$\to$ `S2-338723016` (Source 2) | Shiva Balaji Foods Private Limited | **शिवा बालाजी फूड्स प्राइवेट लिमिटेड** | 1040/30 Garhi Brahman Road Mayur Vihar, Sonepat, Haryana | **1040/30 GARHI BRAHMAN ROAD MAYUR VIHAR, SONEPAT, Haryana** | Non-Latin/Indic script or accent in name; Lexical variation / typo / alias; Address case variation |
| 19 | `S1-138880664`<br>$\to$ `S2-687579513` (Source 2) | Projects Bean Communication Pvt Ltd | **Projects Bean Pvt Ltd Center** | H.No. A-12, Kh. No-42/11/2, Dass Garden Baprola, New Delhi, South West Delhi, Delhi | **Delhi, H.NO. A-12 , KH. NO-42/11/2, DASS GARDEN BAPROLA, NEW DELHI** | Lexical variation / typo / alias; Street/Door number variation (range or unit) |
| 20 | `S1-777528513`<br>$\to$ `S2-200905281` (Source 2) | Gold Food Private Limited | **गोल्ड फूड प्राइवेट लिमिटेड** | B-433, First Floor, G.D. Colony Phase-3, Mayur Vihar, Delhi, East Delhi, Delhi | **B-433/6, FIRST FLOOR, G.D. COLONY PHASE-3, MAYUR VIHAR, EAST DELHI, DELHI** | Non-Latin/Indic script or accent in name; Lexical variation / typo / alias; Street/Door number variation (range or unit) |
| 21 | `S1-386187339`<br>$\to$ `S2-133120494` (Source 2) | VN Association Limited | **VN ATSODNCTAIMN  LIMITED \| www.vnatsodnc.com** | Gujarat, Flat 303, Surat, Roshan Park Appt, Chowki Sheri, Saiyadpura, Surat City | ***[NULL]*** | URL / Domain appended to name; Lexical variation / typo / alias; Target address MISSING (Null) |
| 22 | `S1-73269870`<br>$\to$ `S2-823873728` (Source 2) | Paper Marketing Pvt Ltd | **Veoumbra** | C/O Kashem Ali Khan, Vill, Shibramchak, Ps-Rajapur, Howrah, West Bengal | **C/O KASHEM ALI KHAN, VILL, SHIBRAMCHAK, PS-RAJAPUR, HOWRAH, West Bengal** | Lexical variation / typo / alias; Address case variation |
| 23 | `S1-83816495`<br>$\to$ `S2-934066023` (Source 2) | Black Management Private Limited | **ब्लैक मैनेजमेंट प्राइवेट लिमिटेड** | Vphc-05 Shpra K Vista, Indirapuram Ghaziabad, Ghaziabad, Uttar Pradesh | **VPHC-##05 SHPRA K VISTA, GHAZIABAD, उत्तर प्रदेश** | Non-Latin/Indic script or accent in name; Lexical variation / typo / alias; Street/Door number variation (range or unit) |
| 24 | `S1-306664453`<br>$\to$ `S2-143066677` (Source 2) | Kishore Brothers Private Limited | **KISHORE  8ROTHERS PRIVATE LIMITED** | P No 175/1, 175/2, Kiranmai, Apartments, Mothi Nagar, Rangareddy, Telangana | **P NO 175/1, 175/2, KIRNMAI, APARTMENTS, MOTHI NAGAR, RANGAREDDY, Telangana** | Lexical variation / typo / alias |
| 25 | `S1-685598087`<br>$\to$ `S2-229874129` (Source 2) | Star Properties Private Limited | **स्टार प्रॉपर्टीज प्राइवेट लिमिटेड** | 52/1 Sarai Kale Khan, New Delhi, South Delhi, Delhi | **##52/1 SARAI KALE KHAN, NEW DELHI, SOUTH DELHI, दिल्ली** | Non-Latin/Indic script or accent in name; Lexical variation / typo / alias; Street/Door number variation (range or unit) |
| 26 | `S1-596053169`<br>$\to$ `S2-390541192` (Source 2) | Divine Ventures | **Dr Divine Ventures LLP LLP** | 4E, 4Th Floor, Pioneer Homes, Manthira Apartment, 23A, North Boag Road, T Nagar, Chennai, Tamil Nadu | **#866- 4E, 4TH FLOOR, PIONEER HOMES, MANTHIRA APARTMENT, 23A, NORTH BOAG ROAD, T NAGAR, CHENNAI, Tamil Nadu** | Legal suffix/prefix addition or omission; Street/Door number variation (range or unit) |
| 27 | `S1-4022207`<br>$\to$ `S2-152598253` (Source 2) | Mumbai Aluminium Group | **Mumbai Aluminium** | Maharashtra, Mumbai, Flat No 501 Sarkar Cornerco Op Hsg Soc Veera Desai Road Andheri West, Mumbai | ***[NULL]*** | Legal suffix/prefix addition or omission; Target address MISSING (Null) |
| 28 | `S1-379558062`<br>$\to$ `S2-605760269` (Source 2) | Vocal Educational Society | **Smt Vocal Society Center** | 19/2 33Rd Cross, 2Nd B Main Nr Rto Office, Bangalore North, Bangalore, Karnataka | **##19/2 33RD CROSS, 2ND B MAIN NR RTO OFFICE, BANGALORE NORTH, Karnataka** | Lexical variation / typo / alias; Street/Door number variation (range or unit) |
| 29 | `S1-669461780`<br>$\to$ `S2-674622229` (Source 2) | Suraj Trading Private Limited | **suraj trading private limited** | No 69 (Old No 29), Eldams Road, Teynampet, Chennai, Tamil Nadu | **NO A-69 (OLD NO 29), ELDAMS ROAD, TEYNAMPET, CHENNAI, Tamil Nadu** | Case variation only; Street/Door number variation (range or unit) |
| 30 | `S1-285171170`<br>$\to$ `S2-594484691` (Source 2) | Dynamic Consultants Pvt Ltd | **డైనమిక్ కన్సల్టెంట్స్ ప్రైవేట్ లిమిటెడ్** | 2Nd Floor Dwaraka Heights, Plot No.17 Jubilee Enclave, Madhapur, Hitech City, Hyderabad, Telangana | **2ND FLOOR DWARAKA HEIGHTS, PLOT NO.17 JUBILEE ENCLAVE, MADHAPUR, HITECH CITY, HYDERABAD, Andhra Pradesh** | Non-Latin/Indic script or accent in name; Lexical variation / typo / alias; Street/Door number variation (range or unit) |

---

## 4. Feature Engineering Architecture Implications

Based on these empirical transformation behaviors, our ML feature pipeline must implement the following dedicated feature groups:

### 4.1 Name Similarity Feature Group
| Feature | Rationale & Noise Pattern Handled |
| :--- | :--- |
| `name_token_sort_ratio` | Invariant to word reordering (e.g., `Sree Limited Private Construction` vs `Sree Construction Private Limited`). |
| `name_token_set_ratio` | Invariant to extraneous tokens (e.g. appended URLs `| www.vargaspho.com` or legal affixes). |
| `name_no_legal_levenshtein` | Computes edit distance after legal forms are stripped, preventing mismatch penalties on `LLC` vs `Inc` vs omitted suffix. |
| `name_jaro_winkler` | Weights common prefixes heavily (e.g. `Vargas Phoenix` vs `Vargas Phoenix LLC`). |
| `multilingual_embedding_cosine` | **Critical for Indic scripts**: Cross-lingual dense embeddings (e.g. `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`) match Tamil/Hindi script names with their English phonetic forms where lexical edit distance is 0.0. |
| `phonetic_match_double_metaphone` | Handles English & Indic phonetic alternations (`Sree` vs `Shree`, `Center` vs `Centre`). |

### 4.2 Address Similarity Feature Group
| Feature | Rationale & Noise Pattern Handled |
| :--- | :--- |
| `address_token_set_ratio` | Invariant to address component inversions (`[State], [City], [Street]` vs `[Street], [City], [State]`). Linear Levenshtein fails completely on inverted addresses. |
| `street_number_exact_match` | Checks if street number is identical. |
| `street_number_fuzzy_match` | Handles range extensions (`674` vs `674-678`) and alphanumeric suffixes (`1287` vs `1287-B`). |
| `state_match` | Exact match on standardized 2-letter state / province code (`AL`, `IN`, `Maharashtra`, `Tamil Nadu`). |
| `city_similarity` | Fuzzy matching on city names (`Ardmore` vs `Ardmore Township`, `Jaipur` vs `Jaipur City`). |
| `landmark_similarity` | Compares extracted landmark fields independently so presence/absence of landmark does not distort core street matching. |
| `address_null_indicator` | Explicit binary indicator when S2/S3 address is null, allowing the tree models to learn separate name-only decision branches. |
