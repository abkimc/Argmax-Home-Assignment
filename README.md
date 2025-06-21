# 🥗 Meal Diet Classifier

This repository contains a solution for classifying meals as **vegan** and/or **keto** using a combination of rule-based heuristics, a nutritional database, and a machine learning model — **without relying on pre-labeled data**.

---

## 🧠 Classification Criteria

- **Vegan Meal**: Contains _no animal products_ (e.g., meat, poultry, fish, dairy, eggs, honey, etc.).
- **Keto Meal**: Contains _no single ingredient_ with more than **10g of carbohydrates per 100g**.

---

## 🚀 Features

- **🧹 Ingredient Parsing**  
  A robust parser standardizes messy ingredient strings into clean, interpretable names.

- **🌱 Vegan Classifier**  
  Combines rules, keyword matching, and a transformer-based ML model to detect animal-derived products.

- **🥑 Keto Classifier**  
  Uses rule-based rejection/approval and USDA-based nutrition data with fuzzy matching.

- **⚡ Performance Optimized**  
  Caching and lazy-loading reduce redundant computations.
---

## ⚙️ How It Works

### 🧼 Ingredient Parsing

The `parse_ingredient()` utility is the first step. It:

- Converts to lowercase
- Removes parentheticals
- Strips units, numbers, fractions (e.g., `2 1/2 cups`, `100g`)
- Removes adjectives (e.g., `fresh`, `chopped`)
- Normalizes spacing

> Example:  
> `"2 1/2 cups (12.5 oz) sifted all-purpose flour, for dusting"`  
> ⟶ `"all-purpose flour"`

---

### 🌱 1. Vegan Classifier

A recipe is **vegan only if _all ingredients_ are vegan**.

#### 🔄 Classification Flow:

1. **Cache Lookup**  
   Uses `VEGAN_CACHE` to avoid repeated computation.

2. **Rule-Based Heuristics (Fast Path)**  
   - 🛡️ Edge Case Handling: `"eggless"` ≠ `"egg"`
   - 🥛 Vegan Prefix Pass: `"soy milk"`, `"cashew cheese"` ⟶ safe
   - 🚫 Non-Vegan Keywords: e.g., `chicken`, `milk`, `cheese`
   - ✅ Always-Vegan Keywords: e.g., `flour`, `salt`, `olive oil`

3. **🤖 ML Fallback**  
   If rules don’t resolve the classification:
   - Uses [`nisuga/food_type_classification_model`](https://huggingface.co/nisuga/food_type_classification_model)
   - Predicts `'PLANT_BASED'` vs. `'ANIMAL_BASED'`
   - If prediction fails → defaults to **non-vegan**

4. **🗃️ Cache Update**  
   Result is stored for future reuse.

---

### 🥑 2. Keto Classifier

A recipe is **keto only if _every ingredient_ has ≤10g carbs per 100g**.

#### 🔄 Classification Flow:

1. **Keyword Checks**
   - 🚫 `NON_KETO_KEYWORDS`: e.g., `sugar`, `bread`, `rice` ⟶ immediate reject
   - ✅ `MANUAL_KETO_OVERRIDES`: e.g., `beef`, `avocado`, `olive oil` ⟶ fast pass

2. **🍽️ Nutrition Database Lookup**
   - Source: `data/nutrition_database.csv` - a modified version of the sr legacy data base (https://www.ars.usda.gov/northeast-area/beltsville-md-bhnrc/beltsville-human-nutrition-research-center/methods-and-application-of-food-composition-laboratory/mafcl-site-pages/sr-legacy-nutrient-search/)
   - Uses `fuzz.WRatio` from [`thefuzz`](https://pypi.org/project/thefuzz/) to fuzzy match ingredients
   - Requires a confidence score > **92%**

3. **❗Safe Default**
   - If no confident match is found, assign `1001g carbs` to force rejection

4. **💾 Cache Lookup**
   - Results stored in `SEARCH_CACHE` to improve speed

---
   
# 🥑 Search By Ingredients Challenge
![Argmax](https://argmaxml.com/wp-content/uploads/2024/04/Argmax_logo_inline.svg)

🗓 Submission Deadline: June 30th, 2025


🎥 Please watch [this explainer video](https://youtu.be/rfdaZXseRro) to understand the task.

## 👋 Who Is This Repo For?

[Argmax](https://www.argmaxml.com) is hiring Junior Data scientists in Israel (TLV) and the United States (NYC area).
This repo is meant to be a the first step in the process and it will set the stage for the interview.

The data is taken from a real-life scenario, and it reflects the type of work you will do at Argmax.


## 💼 About the Position

Argmax is a boutique consulting firm specializing in personalized search and recommendation systems. We work with medium to large-scale clients across retail, advertising, healthcare, and finance.

We use tools like large language models, vector databases, and behavioral data to build personalization systems that actually deliver results.

We're looking for candidates who are:

-	✅ Proficient in Python
-	🔍 Naturally curious
-	🧠 Able to perform independent research

This challenge is designed to simulate the type of problems you'll tackle with us, and it applies to positions in both our:
-	🇮🇱 Ramat Gan, Israel office
-	🇺🇸 North Bergen County, New Jersey office

## 🎥 Past Project Talks

1. [Uri's talk on Persona based evaluation with large language models](https://www.youtube.com/watch?v=44--JTG0aMg)
1. [Benjamin Kempinski on offline metrics](https://www.youtube.com/watch?v=5OPa2RYL5VI)
1. [Daniel Hen & Uri Goren on pricing with contextual bandits](https://www.youtube.com/watch?v=IJtNBbINKbI)
1. [Eitan Zimmerman's talk on visual feed reranking](https://www.youtube.com/watch?v=q4uF8nF5SWk)

## 🚀 Getting Started

### 🛠️ Setup

1.	Make sure Docker is installed on your machine.
1.	Run the following in your terminal:  `docker compose build` and  `docker compose up -d`
1. Open your browser and go to [localhost:8888](http://localhost:8888)
1. Follow the instructions in the [task.ipynb](https://github.com/argmaxml/search_by_ingredients/blob/master/nb/src/task.ipynb) notebook

### 📬 Submission Instructions

1. **Clone** this repository into a **private GitHub repo** under your own account.
1. **Invite** [argmax2025](https://github.com/argmax2025) as a collaborator.
1. **Implement** the missing parts in the codebase.
1. Once done, fill in the application form:
1.1. [US Application Form](https://forms.clickup.com/25655193/f/rexwt-1832/L0YE9OKG2FQIC3AYRR) 
1.1 [IL Application Form](https://forms.clickup.com/25655193/f/rexwt-1812/IP26WXR9X4P6I4LGQ6)
1. We'll reach out to you after reviewing your submission.

## 🧪 The Interview Process
### 🧑‍💻 Hands-On Technical Interview (July 2025)

1.	A 3-hour live coding session focused on your submitted solution.
1.	You'll be asked to extend, modify, and explain parts of the codebase.
1.	Please ensure you're in a quiet space with a workstation capable of running your solution.

### 🏢 On-Site Interview (August-September 2025)

1. A non-technical, in-person meeting at our offices in Ramat Gan or New Jersey.
1. We’ll get to know you and discuss your goals.
1. Successful candidates will receive offers around late August or early September.

## ❓ Still Have Questions?

Feel free to mail us at [challenge25@argmaxml.com](mailto:challenge25@argmaxml.com)

