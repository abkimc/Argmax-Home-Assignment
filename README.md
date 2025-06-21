# 🥑 Keto Classifier & 🌱 Vegan Classifier

A multi-layered classification system for dietary labels based on ingredient-level heuristics and machine learning.

---

## 🥑 Keto Classifier

<h3 style="color: #3b82f6;">Objective</h3>

> _"Contains no ingredients with more than 10g of carbohydrates per 100g serving."_

### 🚦 Process Overview: Fast-Fail System

1. **Manual Keto Override (Fast Pass)**  
   ~30 always-keto ingredients (e.g. butter, beef, avocado)

2. **Manual Non-Keto Rejection (Fast Fail)**  
   ~100 non-keto ingredients (e.g. sugar, flour, potato)

3. **Database Fallback (Fuzzy Matching)**  
   Query USDA database with `thefuzz` (WRatio > 90)

4. **Default to Non-Keto**  
   _Guilty until proven innocent_

> ✅ A recipe is only classified as Keto if **every** ingredient passes **all** checks.

---

## 🌱 Vegan Classifier

<h3 style="color: #16a34a;">Objective</h3>

> _"Absolutely no animal products."_

### 🧠 Process Overview: Hybrid Heuristic + ML

1. **Vegan Prefix Pass**  
   Identify vegan alternatives (e.g. almond milk, peanut butter)

2. **Non-Vegan Rejection**  
   Screen against 200+ non-vegan ingredients (meat, dairy, seafood)

3. **Definitive Vegan Pass**  
   Recognize common vegan staples (e.g. water, flour)

4. **ML Model Fallback**  
   Transformer model for ambiguous ingredients  
   `argmaxinc/deberta-v3-base-plant-animal-based` (HuggingFace)

> ✅ A recipe is only classified as Vegan if **every** ingredient passes **all** checks.

---

## ⚙️ Shared Challenges & Solutions

| Challenge | Solution |
|----------|----------|
| **Messy ingredients** | `robust_ingredient_splitter` and `inflect` for de-pluralization |
| **Over-permissiveness** | Expanded keyword lists and stricter defaults |
| **Bias on known data** | Ongoing keyword tuning and generalization strategies |
| **Transformer performance** | Rule-based filters + caching to reduce inference calls |
| **Semantic ambiguity** | Prefix-based rules to detect edge cases like _peanut butter_ |

---

## 🔄 Flowcharts (Simplified)

**Keto Flow:**  
`Ingredient → Fast Pass → Fast Fail → Fuzzy Match → Pass/Fail`

**Vegan Flow:**  
`Ingredient → Rule-Based Checks → ML Fallback → Cache Result → Pass/Fail`

---

## 🛠 Technologies

- Python 3.x
- `thefuzz`
- USDA food database
- Hugging Face Transformers

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

