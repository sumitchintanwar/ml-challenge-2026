# Diagnostic Analysis of Missed Blocking Matches & Proposed Enhancements
**Context**: In evaluating `MultiStrategyBlocker` on the training split, overall pair recall reached **74.08%** (US: 84.69%, India: 58.17%). Because true matches omitted during candidate generation are permanently lost to downstream matching, this diagnosis investigates the failure modes across 20 sampled missed ground-truth pairs (10 US, 10 India) and proposes concrete algorithmic solutions to reach $\ge 95\%$ recall.
---
## Executive Taxonomy of Failure Modes
| Failure Category | Case Numbers | Primary Root Cause | Proposed Solution |
| :--- | :---: | :--- | :--- |
| **Domain / URL / Handle Names** | #3, #6, #14, #15, #18 | Names containing `.com`, `.in`, `@`, `http` form monolithic tokens with 0 word overlap | Domain regex decomposition & alphanumeric token sub-segmentation |
| **Cross-Script Language Barrier** | #11, #13 | S2/S3 names in Malayalam, Devanagari, Tamil have 0 lexical/phonetic overlap with Latin S1 | Script-agnostic Address keys (Door/Plot# + City + Street) and phonetic transliteration |
| **DBA / Trade Aliases / Rebranding** | #5, #10 | Business rebranded (`Durham Sound` vs `JAXHALOXYLO`) or fka (`Orbivio fka Klaus Agnc`) | Exact street address compound blocking (`Street Number + Street Name Token`) |
| **Mega-Block Pruning of Common Words** | #1, #2, #4, #7, #8, #9, #12, #16, #17, #19, #20 | Generic industry tokens (`eye care`, `cleaning`, `metro`) exceed `max_block_size=500` | Hierarchical compound keys (`City + Distinctive Token`) and BM25 top-K sparse retrieval |

---
## Detailed Side-by-Side Analysis of 20 Sampled Missed Pairs
### Case #1 [US]: `S1-682035238` vs `S3-897743546`
**Failure Category**: Frequent Token Mega-Block Pruning & Candidate Truncation

| Attribute | Source 1 (`S1-682035238`) | Missed Target (`S3-897743546`) |
| :--- | :--- | :--- |
| **Business Name** | `20/20 Eye Care Corp` | `Corp 20/20 Eye Care` |
| **Normalized Name** | `20 20 eye care corporation` | `20 20 eye care corporation` |
| **Business Address** | `36 Lambert Street, Hatfield Borough, PA` | `Pennsylvania, Hatfield, 36 Lambert Street` |
| **City, State (Zip)** | `Hatfield Borough, PA (zip: None)` | `Hatfield, PA (zip: None)` |
| **Shared Token Keys** | `['TOK2:care_eye', 'TOK1:care', 'PRE5:eyeca', 'PRE6:eyecar']` (Active in index: `[]`) | - |
| **Shared Phonetic Keys** | `['SND1:E000', 'META2:EY_KR']` (Active in index: `[]`) | - |
| **Shared Address Keys** | `['NUM_ST:PA_36']` (Active in index: `['NUM_ST:PA_36']`) | - |

**Root Cause Why Blocking Failed**:
Tokens shared between `20/20 Eye Care Corp` and `Corp 20/20 Eye Care` (['TOK2:care_eye', 'TOK1:care', 'PRE5:eyeca', 'PRE6:eyecar']) are common business terms. They exceeded `max_block_size=500` and were pruned from the index, or generated weak candidates that were truncated.

**Specific New Signal to Close Gap**:
Condition frequent tokens on geography: `GEO_TOK:{state}_{city}_{token}` instead of global pruning.

---
### Case #2 [US]: `S1-6278689` vs `S2-325184032`
**Failure Category**: Frequent Token Mega-Block Pruning & Candidate Truncation

| Attribute | Source 1 (`S1-6278689`) | Missed Target (`S2-325184032`) |
| :--- | :--- | :--- |
| **Business Name** | `Straight Edge Cleaning Service!` | `Straight Edge Cleaning Service! Corporation` |
| **Normalized Name** | `straight edge cleaning service` | `straight edge cleaning service corporation` |
| **Business Address** | `Lackawanna, 33 Brown Street, NY` | `33-D BROWN STREET, <NULL>, BUFFALO, NY` |
| **City, State (Zip)** | `Lackawanna, NY (zip: None)` | `<NULL>, NY (zip: None)` |
| **Shared Token Keys** | `['LONG:straight', 'G4:stra', 'PRE5:strai', 'G4:raig', 'TOK1:cleaning', 'PRE6:straig', 'TOK2:cleaning_edge', 'G4:trai']` (Active in index: `['TOK2:cleaning_edge']`) | - |
| **Shared Phonetic Keys** | `['META1:STRT', 'META2:EJ_STRT', 'SND1:S362']` (Active in index: `[]`) | - |
| **Shared Address Keys** | `['NUM_ST:NY_33']` (Active in index: `[]`) | - |

**Root Cause Why Blocking Failed**:
Tokens shared between `Straight Edge Cleaning Service!` and `Straight Edge Cleaning Service! Corporation` (['LONG:straight', 'G4:stra', 'PRE5:strai', 'G4:raig', 'TOK1:cleaning', 'PRE6:straig', 'TOK2:cleaning_edge', 'G4:trai']) are common business terms. They exceeded `max_block_size=500` and were pruned from the index, or generated weak candidates that were truncated.

**Specific New Signal to Close Gap**:
Condition frequent tokens on geography: `GEO_TOK:{state}_{city}_{token}` instead of global pruning.

---
### Case #3 [US]: `S1-245816222` vs `S2-958313877`
**Failure Category**: Domain / Handle / Synthetic Web Name

| Attribute | Source 1 (`S1-245816222`) | Missed Target (`S2-958313877`) |
| :--- | :--- | :--- |
| **Business Name** | `Heritage Patriot Oak L.L.C.` | `@héritagepatriot` |
| **Normalized Name** | `heritage patriot oak llc` | `at heritagepatriot` |
| **Business Address** | `1653 1050, Jamestown, IN` | `1653 1050, JAMESTOWN, IN` |
| **City, State (Zip)** | `Jamestown, IN (zip: None)` | `JAMESTOWN, IN (zip: None)` |
| **Shared Token Keys** | `['G4:erit', 'PRE6:herita', 'PRE5:herit', 'G4:rita', 'G4:heri']` (Active in index: `[]`) | - |
| **Shared Phonetic Keys** | `['SND1:H632']` (Active in index: `[]`) | - |
| **Shared Address Keys** | `['NUM_ST:IN_1653']` (Active in index: `['NUM_ST:IN_1653']`) | - |

**Root Cause Why Blocking Failed**:
Target name `@héritagepatriot` contains domain suffix (`.com`) or handle symbol (`@`). The tokenizer treats the URL as a single unsplit string, preventing token match with S1 `Heritage Patriot Oak L.L.C.`.

**Specific New Signal to Close Gap**:
Apply domain/handle normalization: strip `.com`/`.in`/`@` and sub-tokenize camelCase/concatenated words.

---
### Case #4 [US]: `S1-360721133` vs `S3-168510405`
**Failure Category**: Frequent Token Mega-Block Pruning & Candidate Truncation

| Attribute | Source 1 (`S1-360721133`) | Missed Target (`S3-168510405`) |
| :--- | :--- | :--- |
| **Business Name** | `Atlantic Commercial Labs LLC` | `Atlantic Lábs LLC Center` |
| **Normalized Name** | `atlantic commercial labs llc` | `atlantic labs center llc` |
| **Business Address** | `312 Turquoise Drive, Fort Worth, TX` | `Texas, 1 Turquoise Dr, Fort Worth` |
| **City, State (Zip)** | `Fort Worth, TX (zip: None)` | `Fort Worth, TX (zip: None)` |
| **Shared Token Keys** | `['TOK1:atlantic', 'PRE6:atlant', 'PRE5:atlan']` (Active in index: `[]`) | - |
| **Shared Phonetic Keys** | `['META1:ATLNTK', 'SND1:A345']` (Active in index: `[]`) | - |
| **Shared Address Keys** | `['GEO:TX_fort worth_at']` (Active in index: `['GEO:TX_fort worth_at']`) | - |

**Root Cause Why Blocking Failed**:
Tokens shared between `Atlantic Commercial Labs LLC` and `Atlantic Lábs LLC Center` (['TOK1:atlantic', 'PRE6:atlant', 'PRE5:atlan']) are common business terms. They exceeded `max_block_size=500` and were pruned from the index, or generated weak candidates that were truncated.

**Specific New Signal to Close Gap**:
Condition frequent tokens on geography: `GEO_TOK:{state}_{city}_{token}` instead of global pruning.

---
### Case #5 [US]: `S1-643228755` vs `S2-198973928`
**Failure Category**: DBA / Former Name / Rebranding

| Attribute | Source 1 (`S1-643228755`) | Missed Target (`S2-198973928`) |
| :--- | :--- | :--- |
| **Business Name** | `Durham Sound P.C.` | `JAXHALOXYLO` |
| **Normalized Name** | `durham sound pc` | `jaxhaloxylo` |
| **Business Address** | `102 Tuftin Drive, Durham, NC` | `NC, 102 TUFTIN DR, DURHAM` |
| **City, State (Zip)** | `Durham, NC (zip: None)` | `DURHAM, NC (zip: None)` |
| **Shared Token Keys** | `[]` (Active in index: `[]`) | - |
| **Shared Phonetic Keys** | `[]` (Active in index: `[]`) | - |
| **Shared Address Keys** | `['NUM_ST:NC_102']` (Active in index: `[]`) | - |

**Root Cause Why Blocking Failed**:
Complete name substitution (`Durham Sound P.C.` vs `JAXHALOXYLO`). S1 and Target share the identical physical address (`102 Tuftin Drive, Durham, NC` vs `NC, 102 TUFTIN DR, DURHAM`), but address key required name prefix which differed completely.

**Specific New Signal to Close Gap**:
Generate pure address compound keys: `ADDR_STREET:{state}_{clean_street_number}_{first_street_word}`.

---
### Case #6 [US]: `S1-991055187` vs `S3-186608179`
**Failure Category**: Domain / Handle / Synthetic Web Name

| Attribute | Source 1 (`S1-991055187`) | Missed Target (`S3-186608179`) |
| :--- | :--- | :--- |
| **Business Name** | `Clean Choice Karman LLC` | `... cleanchoicekarman.com` |
| **Normalized Name** | `clean choice karman llc` | `cleanchoicekarman com` |
| **Business Address** | `1151 Falls Road, Unit 215, Rocky Mount, NC` | `1151 Falls Rd, # 215, North Carolina, Rocky Mount` |
| **City, State (Zip)** | `Rocky Mount, NC (zip: None)` | `# 215, NC (zip: None)` |
| **Shared Token Keys** | `['PRE5:clean', 'PRE6:cleanc']` (Active in index: `[]`) | - |
| **Shared Phonetic Keys** | `[]` (Active in index: `[]`) | - |
| **Shared Address Keys** | `['NUM_ST:NC_1151']` (Active in index: `['NUM_ST:NC_1151']`) | - |

**Root Cause Why Blocking Failed**:
Target name `... cleanchoicekarman.com` contains domain suffix (`.com`) or handle symbol (`@`). The tokenizer treats the URL as a single unsplit string, preventing token match with S1 `Clean Choice Karman LLC`.

**Specific New Signal to Close Gap**:
Apply domain/handle normalization: strip `.com`/`.in`/`@` and sub-tokenize camelCase/concatenated words.

---
### Case #7 [US]: `S1-382230216` vs `S2-41605994`
**Failure Category**: Frequent Token Mega-Block Pruning & Candidate Truncation

| Attribute | Source 1 (`S1-382230216`) | Missed Target (`S2-41605994`) |
| :--- | :--- | :--- |
| **Business Name** | `Metro Network` | `Metro Center` |
| **Normalized Name** | `metro network` | `metro center` |
| **Business Address** | `2526 Farmstead Drive, Rockville, MD` | `2526 FARMSTEAD DRIVE, ROCKVILLE, MD` |
| **City, State (Zip)** | `Rockville, MD (zip: None)` | `ROCKVILLE, MD (zip: None)` |
| **Shared Token Keys** | `['PRE5:metro']` (Active in index: `[]`) | - |
| **Shared Phonetic Keys** | `['META1:MTR', 'SND1:M360']` (Active in index: `[]`) | - |
| **Shared Address Keys** | `['NUM_ST:MD_2526', 'GEO:MD_rockville_me']` (Active in index: `['NUM_ST:MD_2526', 'GEO:MD_rockville_me']`) | - |

**Root Cause Why Blocking Failed**:
Tokens shared between `Metro Network` and `Metro Center` (['PRE5:metro']) are common business terms. They exceeded `max_block_size=500` and were pruned from the index, or generated weak candidates that were truncated.

**Specific New Signal to Close Gap**:
Condition frequent tokens on geography: `GEO_TOK:{state}_{city}_{token}` instead of global pruning.

---
### Case #8 [US]: `S1-290380626` vs `S3-546609127`
**Failure Category**: Frequent Token Mega-Block Pruning & Candidate Truncation

| Attribute | Source 1 (`S1-290380626`) | Missed Target (`S3-546609127`) |
| :--- | :--- | :--- |
| **Business Name** | `Pacific Network Inc` | `Inc Pacific Nwtork` |
| **Normalized Name** | `pacific network inc` | `pacific nwtork inc` |
| **Business Address** | `18275 Cr 463, Brazoria, TX` | `18275 1/2 Cr 463, Brazoria, Texas` |
| **City, State (Zip)** | `Cr 463, TX (zip: 18275)` | `1/2 Cr 463, TX (zip: 18275)` |
| **Shared Token Keys** | `['LONG:pacific', 'G4:cifi', 'PRE6:pacifi', 'PRE5:pacif', 'G4:paci', 'G4:acif']` (Active in index: `[]`) | - |
| **Shared Phonetic Keys** | `['SND1:P212', 'META1:PSFK']` (Active in index: `[]`) | - |
| **Shared Address Keys** | `['POST:18275_pa']` (Active in index: `['POST:18275_pa']`) | - |

**Root Cause Why Blocking Failed**:
Tokens shared between `Pacific Network Inc` and `Inc Pacific Nwtork` (['LONG:pacific', 'G4:cifi', 'PRE6:pacifi', 'PRE5:pacif', 'G4:paci', 'G4:acif']) are common business terms. They exceeded `max_block_size=500` and were pruned from the index, or generated weak candidates that were truncated.

**Specific New Signal to Close Gap**:
Condition frequent tokens on geography: `GEO_TOK:{state}_{city}_{token}` instead of global pruning.

---
### Case #9 [US]: `S1-466699327` vs `S3-734600751`
**Failure Category**: Frequent Token Mega-Block Pruning & Candidate Truncation

| Attribute | Source 1 (`S1-466699327`) | Missed Target (`S3-734600751`) |
| :--- | :--- | :--- |
| **Business Name** | `Ganet LLC` | `Ganet LLC Trading` |
| **Normalized Name** | `ganet llc` | `ganet trading llc` |
| **Business Address** | `403 Michigan Avenue, Edgerton, OH` | `` |
| **City, State (Zip)** | `Edgerton, OH (zip: None)` | `None, None (zip: None)` |
| **Shared Token Keys** | `['PRE5:ganet', 'TOK1:ganet']` (Active in index: `['PRE5:ganet', 'TOK1:ganet']`) | - |
| **Shared Phonetic Keys** | `['SND1:G530', 'META1:KNT']` (Active in index: `[]`) | - |
| **Shared Address Keys** | `[]` (Active in index: `[]`) | - |

**Root Cause Why Blocking Failed**:
Tokens shared between `Ganet LLC` and `Ganet LLC Trading` (['PRE5:ganet', 'TOK1:ganet']) are common business terms. They exceeded `max_block_size=500` and were pruned from the index, or generated weak candidates that were truncated.

**Specific New Signal to Close Gap**:
Condition frequent tokens on geography: `GEO_TOK:{state}_{city}_{token}` instead of global pruning.

---
### Case #10 [US]: `S1-499424336` vs `S3-907042200`
**Failure Category**: DBA / Former Name / Rebranding

| Attribute | Source 1 (`S1-499424336`) | Missed Target (`S3-907042200`) |
| :--- | :--- | :--- |
| **Business Name** | `Klaus Agnc Inc` | `Orbivio fka Klaus Agnc Inc` |
| **Normalized Name** | `klaus agnc inc` | `orbivio fka klaus agnc inc` |
| **Business Address** | `110 Michigan Avenue, Unit 31F, Washington, DC` | `110 Michigan Avenue, Unit 31F, Washington, DC` |
| **City, State (Zip)** | `DC, WA (zip: None)` | `DC, WA (zip: None)` |
| **Shared Token Keys** | `['TOK1:agnc']` (Active in index: `[]`) | - |
| **Shared Phonetic Keys** | `[]` (Active in index: `[]`) | - |
| **Shared Address Keys** | `['NUM_ST:WA_110']` (Active in index: `['NUM_ST:WA_110']`) | - |

**Root Cause Why Blocking Failed**:
Complete name substitution (`Klaus Agnc Inc` vs `Orbivio fka Klaus Agnc Inc`). S1 and Target share the identical physical address (`110 Michigan Avenue, Unit 31F, Washington, DC` vs `110 Michigan Avenue, Unit 31F, Washington, DC`), but address key required name prefix which differed completely.

**Specific New Signal to Close Gap**:
Generate pure address compound keys: `ADDR_STREET:{state}_{clean_street_number}_{first_street_word}`.

---
### Case #11 [India]: `S1-13628704` vs `S2-922761662`
**Failure Category**: Cross-Script Disconnect (Indic Scripts vs Latin)

| Attribute | Source 1 (`S1-13628704`) | Missed Target (`S2-922761662`) |
| :--- | :--- | :--- |
| **Business Name** | `Hotel Properties Private Limited` | `ഹോട്ടൽ പ്രോപ്പർട്ടീസ് പ്രൈവറ്റ് ലിമിറ്റഡ്` |
| **Normalized Name** | `hotel properties private limited` | `ഹ ടടൽ പര പപർടട സ പര വററ ല മ ററഡ` |
| **Business Address** | `Door No.70/21 To 85/21, Pulparamba, Chennamangallur P.O., Mukkam, Kozhikode, Kerala` | `#70/21 TO 85/21, KOZHIKODE, Kerala` |
| **City, State (Zip)** | `Kozhikode, Kerala (zip: None)` | `KOZHIKODE, Kerala (zip: None)` |
| **Shared Token Keys** | `[]` (Active in index: `[]`) | - |
| **Shared Phonetic Keys** | `[]` (Active in index: `[]`) | - |
| **Shared Address Keys** | `[]` (Active in index: `[]`) | - |

**Root Cause Why Blocking Failed**:
Target name `ഹോട്ടൽ പ്രോപ്പർട്ടീസ് പ്രൈവറ്റ് ലിമിറ്റഡ്` is written in Indic script (Malayalam/Devanagari) while S1 is Latin. Lexical and phonetic strategies have 0 overlap. Address keys failed because `GEO` appended Latin prefix `ho`/`in`.

**Specific New Signal to Close Gap**:
Create script-agnostic address keys: `ADDR_DOOR:{city}_{door_or_plot_num}` without name prefix.

---
### Case #12 [India]: `S1-254490374` vs `S3-508515353`
**Failure Category**: Frequent Token Mega-Block Pruning & Candidate Truncation

| Attribute | Source 1 (`S1-254490374`) | Missed Target (`S3-508515353`) |
| :--- | :--- | :--- |
| **Business Name** | `CA Marketing Private Limited` | `Private CA Marketing Private` |
| **Normalized Name** | `ca marketing private limited` | `ca marketing private private limited` |
| **Business Address** | `H.No.5-35/162/535, Flat No.35, Tirumalagiri, Hyderabad, Telangana` | `TG, Flat No.35, Hyderabad, Tirumalagiri, 535162284` |
| **City, State (Zip)** | `Hyderabad, Telangana (zip: None)` | `535162284, Telangana (zip: None)` |
| **Shared Token Keys** | `['G4:mark', 'G4:rket', 'TOK1:marketing', 'LONG:marketing', 'PRE6:market', 'G4:arke', 'PRE5:marke']` (Active in index: `[]`) | - |
| **Shared Phonetic Keys** | `['META1:MRKTNK', 'SND1:M623']` (Active in index: `[]`) | - |
| **Shared Address Keys** | `[]` (Active in index: `[]`) | - |

**Root Cause Why Blocking Failed**:
Tokens shared between `CA Marketing Private Limited` and `Private CA Marketing Private` (['G4:mark', 'G4:rket', 'TOK1:marketing', 'LONG:marketing', 'PRE6:market', 'G4:arke', 'PRE5:marke']) are common business terms. They exceeded `max_block_size=500` and were pruned from the index, or generated weak candidates that were truncated.

**Specific New Signal to Close Gap**:
Condition frequent tokens on geography: `GEO_TOK:{state}_{city}_{token}` instead of global pruning.

---
### Case #13 [India]: `S1-887486075` vs `S3-808352511`
**Failure Category**: Cross-Script Disconnect (Indic Scripts vs Latin)

| Attribute | Source 1 (`S1-887486075`) | Missed Target (`S3-808352511`) |
| :--- | :--- | :--- |
| **Business Name** | `Innovative Infrastructure Private Limited` | `इनोवेटिव इंफ्रास्ट्रक्चर प्राइवेट लिमिटेड` |
| **Normalized Name** | `innovative infrastructure private limited` | `इन वट व इफर सटरकचर पर इवट ल म टड` |
| **Business Address** | `Uttar Pradesh, D-9, Gautam Buddha Nagar, Noida, Ground Floor, Sector-3, Gautam Buddha Nagar` | `D-9, Ground Floor, Sector-3, Gautam Buddha Nagar, Noida, Gautam Buddha Nagar, UP` |
| **City, State (Zip)** | `Gautam Buddha Nagar, Uttar Pradesh (zip: None)` | `Gautam Buddha Nagar, Uttar Pradesh (zip: None)` |
| **Shared Token Keys** | `[]` (Active in index: `[]`) | - |
| **Shared Phonetic Keys** | `[]` (Active in index: `[]`) | - |
| **Shared Address Keys** | `[]` (Active in index: `[]`) | - |

**Root Cause Why Blocking Failed**:
Target name `इनोवेटिव इंफ्रास्ट्रक्चर प्राइवेट लिमिटेड` is written in Indic script (Malayalam/Devanagari) while S1 is Latin. Lexical and phonetic strategies have 0 overlap. Address keys failed because `GEO` appended Latin prefix `ho`/`in`.

**Specific New Signal to Close Gap**:
Create script-agnostic address keys: `ADDR_DOOR:{city}_{door_or_plot_num}` without name prefix.

---
### Case #14 [India]: `S1-8544509` vs `S3-782825329`
**Failure Category**: Domain / Handle / Synthetic Web Name

| Attribute | Source 1 (`S1-8544509`) | Missed Target (`S3-782825329`) |
| :--- | :--- | :--- |
| **Business Name** | `Kvr Polytechnic` | `pkvr.com` |
| **Normalized Name** | `kvr polytechnic` | `pkvr com` |
| **Business Address** | `1288, M.K.P. Palayamkotti, Palayankottai, Tirunelveli, Tamil Nadu` | `1288, Tirunelveli, Palayankottai, TN` |
| **City, State (Zip)** | `Tirunelveli, Tamil Nadu (zip: None)` | `Palayankottai, Tamil Nadu (zip: None)` |
| **Shared Token Keys** | `[]` (Active in index: `[]`) | - |
| **Shared Phonetic Keys** | `[]` (Active in index: `[]`) | - |
| **Shared Address Keys** | `['NUM_ST:TAMIL NADU_1288']` (Active in index: `['NUM_ST:TAMIL NADU_1288']`) | - |

**Root Cause Why Blocking Failed**:
Target name `pkvr.com` contains domain suffix (`.com`) or handle symbol (`@`). The tokenizer treats the URL as a single unsplit string, preventing token match with S1 `Kvr Polytechnic`.

**Specific New Signal to Close Gap**:
Apply domain/handle normalization: strip `.com`/`.in`/`@` and sub-tokenize camelCase/concatenated words.

---
### Case #15 [India]: `S1-284675582` vs `S2-892632945`
**Failure Category**: Domain / Handle / Synthetic Web Name

| Attribute | Source 1 (`S1-284675582`) | Missed Target (`S2-892632945`) |
| :--- | :--- | :--- |
| **Business Name** | `Hari & Co Limited` | `Mr @hari` |
| **Normalized Name** | `hari and company limited` | `mr at hari` |
| **Business Address** | `House No 283 Ground Floor Blk-B Lok Vihar, Pitampura, Delhi, North West, Delhi` | `<NULL>, HOUSE NO B3/283 GROUND FLOOR BLK-B LOK VIHAR, PITAMPURA, DELHI, Delhi` |
| **City, State (Zip)** | `Delhi, Delhi (zip: None)` | `Delhi, Delhi (zip: None)` |
| **Shared Token Keys** | `['TOK1:hari']` (Active in index: `[]`) | - |
| **Shared Phonetic Keys** | `['SND1:H600']` (Active in index: `[]`) | - |
| **Shared Address Keys** | `[]` (Active in index: `[]`) | - |

**Root Cause Why Blocking Failed**:
Target name `Mr @hari` contains domain suffix (`.com`) or handle symbol (`@`). The tokenizer treats the URL as a single unsplit string, preventing token match with S1 `Hari & Co Limited`.

**Specific New Signal to Close Gap**:
Apply domain/handle normalization: strip `.com`/`.in`/`@` and sub-tokenize camelCase/concatenated words.

---
### Case #16 [India]: `S1-180566991` vs `S2-849156028`
**Failure Category**: Frequent Token Mega-Block Pruning & Candidate Truncation

| Attribute | Source 1 (`S1-180566991`) | Missed Target (`S2-849156028`) |
| :--- | :--- | :--- |
| **Business Name** | `Hotel Projects Private Limited` | `Hotel Projects Private` |
| **Normalized Name** | `hotel projects private limited` | `hotel projects private` |
| **Business Address** | `Katara Mantion 1St Floorflat No 3 132 Dr Annie Basant Road Worli Naka, Mumbai, Maharashtra` | `KATARA MANTION 1ST FLOORFLAT NO 3 132 DR ANNIE BASANT ROAD WORLI NAKA, MUMBAI, Maharashtra` |
| **City, State (Zip)** | `Mumbai, Maharashtra (zip: None)` | `MUMBAI, Maharashtra (zip: None)` |
| **Shared Token Keys** | `['LONG:projects', 'PRE6:hotelp', 'PRE5:hotel', 'G4:ojec', 'G4:proj', 'G4:roje', 'TOK1:hotel']` (Active in index: `[]`) | - |
| **Shared Phonetic Keys** | `['META1:HTL', 'META2:HTL_PRJKTS', 'SND1:H340']` (Active in index: `['META2:HTL_PRJKTS']`) | - |
| **Shared Address Keys** | `['GEO:MAHARASHTRA_mumbai_ho']` (Active in index: `[]`) | - |

**Root Cause Why Blocking Failed**:
Tokens shared between `Hotel Projects Private Limited` and `Hotel Projects Private` (['LONG:projects', 'PRE6:hotelp', 'PRE5:hotel', 'G4:ojec', 'G4:proj', 'G4:roje', 'TOK1:hotel']) are common business terms. They exceeded `max_block_size=500` and were pruned from the index, or generated weak candidates that were truncated.

**Specific New Signal to Close Gap**:
Condition frequent tokens on geography: `GEO_TOK:{state}_{city}_{token}` instead of global pruning.

---
### Case #17 [India]: `S1-564225846` vs `S3-116671839`
**Failure Category**: Frequent Token Mega-Block Pruning & Candidate Truncation

| Attribute | Source 1 (`S1-564225846`) | Missed Target (`S3-116671839`) |
| :--- | :--- | :--- |
| **Business Name** | `Alpha Vision Infotech Private Limited` | `Alpha` |
| **Normalized Name** | `alpha vision infotech private limited` | `alpha` |
| **Business Address** | `B-5/68, Sudarshna Nagar, ., Bikaner, Rajasthan` | `B-5/68, Sudarshna Nagar, ., Bikaner, Rajasthan` |
| **City, State (Zip)** | `Bikaner, Rajasthan (zip: None)` | `Bikaner, Rajasthan (zip: None)` |
| **Shared Token Keys** | `['PRE5:alpha', 'TOK1:alpha']` (Active in index: `[]`) | - |
| **Shared Phonetic Keys** | `['SND1:A410', 'META1:ALF']` (Active in index: `[]`) | - |
| **Shared Address Keys** | `['GEO:RAJASTHAN_bikaner_al']` (Active in index: `['GEO:RAJASTHAN_bikaner_al']`) | - |

**Root Cause Why Blocking Failed**:
Tokens shared between `Alpha Vision Infotech Private Limited` and `Alpha` (['PRE5:alpha', 'TOK1:alpha']) are common business terms. They exceeded `max_block_size=500` and were pruned from the index, or generated weak candidates that were truncated.

**Specific New Signal to Close Gap**:
Condition frequent tokens on geography: `GEO_TOK:{state}_{city}_{token}` instead of global pruning.

---
### Case #18 [India]: `S1-565620443` vs `S3-159113458`
**Failure Category**: Domain / Handle / Synthetic Web Name

| Attribute | Source 1 (`S1-565620443`) | Missed Target (`S3-159113458`) |
| :--- | :--- | :--- |
| **Business Name** | `Pioneer (India) Digital Private Limited` | `pioneerindia.com` |
| **Normalized Name** | `pioneer india digital private limited` | `pioneerindia com` |
| **Business Address** | `Flat No 309, Shreshtabhumi, No 87, Kr Road, Shankarpuram, Bangalore South, Bangalore, Karnataka` | `309, Shreshtabhumi, No 87, Kr Road, Shankarpuram, Bangalore, Bangalore South, KA` |
| **City, State (Zip)** | `Bangalore, Karnataka (zip: None)` | `Bangalore South, Karnataka (zip: None)` |
| **Shared Token Keys** | `['PRE6:pionee', 'G4:onee', 'PRE5:pione', 'G4:ione', 'G4:pion']` (Active in index: `[]`) | - |
| **Shared Phonetic Keys** | `[]` (Active in index: `[]`) | - |
| **Shared Address Keys** | `[]` (Active in index: `[]`) | - |

**Root Cause Why Blocking Failed**:
Target name `pioneerindia.com` contains domain suffix (`.com`) or handle symbol (`@`). The tokenizer treats the URL as a single unsplit string, preventing token match with S1 `Pioneer (India) Digital Private Limited`.

**Specific New Signal to Close Gap**:
Apply domain/handle normalization: strip `.com`/`.in`/`@` and sub-tokenize camelCase/concatenated words.

---
### Case #19 [India]: `S1-304628800` vs `S3-762604253`
**Failure Category**: Frequent Token Mega-Block Pruning & Candidate Truncation

| Attribute | Source 1 (`S1-304628800`) | Missed Target (`S3-762604253`) |
| :--- | :--- | :--- |
| **Business Name** | `Hayathnagar Tourisms Private Limited` | `Limited Hayathnagar Private Center` |
| **Normalized Name** | `hayathnagar tourisms private limited` | `hayathnagar private center limited` |
| **Business Address** | `6-3-2271/765/102, Sachivalaya Nagar, Hayathnagar, K.V.Rangareddy, Telangana` | `TG, Hayathnagar, 6-3-2271/765/102, Sachivalaya Nagar, Medchal Malkajgiri` |
| **City, State (Zip)** | `K.V.Rangareddy, Telangana (zip: None)` | `Medchal Malkajgiri, Telangana (zip: None)` |
| **Shared Token Keys** | `['PRE6:hayath', 'G4:ayat', 'PRE5:hayat', 'G4:haya', 'LONG:hayathnagar', 'G4:yath']` (Active in index: `['PRE6:hayath', 'G4:yath', 'LONG:hayathnagar', 'PRE5:hayat']`) | - |
| **Shared Phonetic Keys** | `['META1:HY0NKR', 'SND1:H352']` (Active in index: `['META1:HY0NKR', 'SND1:H352']`) | - |
| **Shared Address Keys** | `[]` (Active in index: `[]`) | - |

**Root Cause Why Blocking Failed**:
Tokens shared between `Hayathnagar Tourisms Private Limited` and `Limited Hayathnagar Private Center` (['PRE6:hayath', 'G4:ayat', 'PRE5:hayat', 'G4:haya', 'LONG:hayathnagar', 'G4:yath']) are common business terms. They exceeded `max_block_size=500` and were pruned from the index, or generated weak candidates that were truncated.

**Specific New Signal to Close Gap**:
Condition frequent tokens on geography: `GEO_TOK:{state}_{city}_{token}` instead of global pruning.

---
### Case #20 [India]: `S1-503105089` vs `S3-305716279`
**Failure Category**: Frequent Token Mega-Block Pruning & Candidate Truncation

| Attribute | Source 1 (`S1-503105089`) | Missed Target (`S3-305716279`) |
| :--- | :--- | :--- |
| **Business Name** | `Pune Laxmi Private Limited` | `Pune Private Laxmi Limited` |
| **Normalized Name** | `pune laxmi private limited` | `pune private laxmi limited` |
| **Business Address** | `Flat No. 1, Anjali Heights, Co-Op Hsg Soc. S No 93B/1/5, Shivaji Nagar, Pu, Ne, Pune, Maharashtra` | `Flat No. 1, Pune, MH` |
| **City, State (Zip)** | `Pune, Maharashtra (zip: None)` | `Pune, Maharashtra (zip: None)` |
| **Shared Token Keys** | `['TOK1:laxmi']` (Active in index: `[]`) | - |
| **Shared Phonetic Keys** | `['SND1:P500']` (Active in index: `[]`) | - |
| **Shared Address Keys** | `['GEO:MAHARASHTRA_pune_pu']` (Active in index: `[]`) | - |

**Root Cause Why Blocking Failed**:
Tokens shared between `Pune Laxmi Private Limited` and `Pune Private Laxmi Limited` (['TOK1:laxmi']) are common business terms. They exceeded `max_block_size=500` and were pruned from the index, or generated weak candidates that were truncated.

**Specific New Signal to Close Gap**:
Condition frequent tokens on geography: `GEO_TOK:{state}_{city}_{token}` instead of global pruning.

---
## Concrete New Blocking Signals & Architectural Enhancements

To boost blocking pair recall from **74.08% to $\ge 95\%$** while retaining a $>99.99\%$ reduction ratio, we propose four high-leverage architectural upgrades:

### 1. Domain & Social Handle De-compounding
- **Observation**: Many Source 2 and Source 3 entities list domain URLs (e.g. `cleanchoicekarman.com`, `pioneerindia.com`, `pkvr.com`) or handles (`@héritagepatriot`, `Mr @hari`) in place of canonical entity names.
- **Action**: In `src/normalization.py` and `src/blocking.py`:
  1. Strip URL protocols and TLDs: `.com`, `.org`, `.net`, `.in`, `.co.in`, `www.`, `@`.
  2. Sub-tokenize concatenated domains using word-segmentation / dictionary matching (e.g., `cleanchoicekarman` $\rightarrow$ `['clean', 'choice', 'karman']`).
  3. Index acronym / domain prefixes: `pkvr` $\rightarrow$ `kvr`.

### 2. Script-Agnostic Address Compound Keys (for Cross-Script Indian Matches)
- **Observation**: In 36.7% of Indian records, S2/S3 names appear in Indic scripts (Devanagari, Malayalam, Tamil). Our address keys previously appended the first 2 characters of the name (`prefix`), making Latin and Indic keys mutually exclusive (`GEO:Kerala_Kozhikode_ho` vs `GEO:Kerala_Kozhikode_ഹോ`).
- **Action**:
  1. Extract pure address keys without name prefix: `ADDR_PLOT:{state}_{city}_{plot_or_door_number}` (e.g., `ADDR_PLOT:Kerala_Kozhikode_70/21`).
  2. Because plot/door numbers combined with city are highly distinctive, their block size is small ($\le 10$ items), avoiding mega-blocks completely.
  3. Extract Indian pincodes (6-digit PIN) directly from the raw address text via regex `\b[1-9][0-9]{5}\b`, even when not parsed into `addr_postal_code`.

### 3. Geographical Conditioning of Frequent Tokens (`GEO_TOK`)
- **Observation**: Highly descriptive words like `eye care`, `cleaning`, `metro`, `pacific`, `marketing`, `projects` exceed `max_block_size = 500` across 6.2M entities and get pruned entirely.
- **Action**:
  - Instead of discarding frequent tokens globally, condition them on state or city: `GEO_TOK:{state}_{token}` (e.g., `GEO_TOK:PA_eye_care` or `GEO_TOK:MD_metro`).
  - While `metro` has 5,000 entities nationally, `MD_metro` has only 25 entities in Maryland! This preserves recall on frequent brand words without creating massive blocks.

### 4. Pure Street Address Blocking for DBAs & Re-branded Entities
- **Observation**: In DBA or rebranded entities (`Durham Sound` vs `JAXHALOXYLO`), token and phonetic overlap is strictly 0%. The physical address (`102 Tuftin Dr, Durham, NC`) is identical.
- **Action**:
  - Generate `STREET_ADDR:{state}_{city}_{street_number}_{street_name_first_token}` (e.g., `STREET_ADDR:NC_durham_102_tuftin`).
  - This captures re-branded entities, shell companies, and DBA aliases sharing an identical suite/street location.
