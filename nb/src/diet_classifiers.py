import json
import sys
from argparse import ArgumentParser
from typing import List
from time import time
import pandas as pd
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re
from typing import Set
from typing import List, Dict, Any, Set, Optional
from thefuzz import process, fuzz
from typing import Dict, Any, Optional, List
import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
import sys
import ast

try:
    from sklearn.metrics import classification_report
except ImportError:
    # sklearn is optional
    def classification_report(y, y_pred):
        print("sklearn is not installed, skipping classification report")


# ==============================================================================
#  Vegan classifier
# ==============================================================================

UNITS: Set[str] = {
    "c", "cup", "cups",
    "g", "gram", "grams",
    "kg", "kilogram", "kilograms",
    "l", "liter", "liters",
    "lb", "lbs", "pound", "pounds",
    "ml", "milliliter", "milliliters",
    "oz", "ounce", "ounces",
    "pinch", "pinches",
    "splash", "splashes",
    "sprig", "sprigs",
    "t", "tsp", "teaspoon", "teaspoons",
    "T", "tbsp", "tablespoon", "tablespoons",
    "can", "cans",
    "clove", "cloves",
    "dash", "dashes",
    "drizzle",
    "drop", "drops",
    "gallon", "gallons",
    "handful", "handfuls",
    "head", "heads",
    "package", "packages",
    "packet", "packets",
    "pint", "pints",
    "quart", "quarts",
    "scoop", "scoops",
    "sheet", "sheets",
    "slice", "slices",
    "stalk", "stalks",
    "stick", "sticks",
    "strip", "strips",
}
# A comprehensive and categorized set of non-essential words found in ingredient lists.
# The purpose of this set is to remove these words from an ingredient string
# to help isolate the core, identifiable name of the food item.
DESCRIPTORS: Set[str] = {
    # --- Preparation & Actions ---
    'beaten', 'blanched', 'boiled', 'braised', 'brewed', 'brined', 'broken',
    'charred', 'chilled', 'chopped', 'clarified', 'coarsely', 'crumbled', 'crushed',
    'cubed', 'cut', 'deboned', 'deglazed', 'deseeded', 'deveined', 'diced',
    'dissolved', 'divided', 'drained', 'finely', 'flaked', 'folded', 'grated',
    'grilled', 'ground', 'halved', 'heated', 'hulled', 'husked', 'infused',
    'julienned', 'juiced', 'kneaded', 'marinated', 'mashed', 'melted', 'minced',
    'mixed', 'parboiled', 'patted', 'peeled', 'pitted', 'poached', 'pounded',
    'prepared', 'pressed', 'pureed', 'quartered', 'rinsed', 'roasted', 'rolled',
    'roughly', 'scalded', 'scored', 'scrubbed', 'seared', 'seeded', 'segmented',
    'shaved', 'shredded', 'shucked', 'sifted', 'skewered', 'sliced', 'slivered',
    'smashed', 'soaked', 'softened', 'squeezed', 'steamed', 'stemmed', 'stewed',
    'strained', 'stuffed', 'thawed', 'thinly', 'tied', 'toasted', 'torn', 'trimmed',
    'whisked', 'zested',

    # --- State, Condition & Temperature ---
    'canned', 'cold', 'condensed', 'cooked', 'cooled', 'cored', 'creamed', 'cured',
    'defrosted', 'dried', 'fermented', 'firmly', 'fresh', 'freshly', 'frozen',
    'hard', 'hot', 'instant', 'jarred', 'lean', 'leftover', 'light', 'lukewarm',
    'optional', 'pasteurized', 'powdered', 'preserved', 'raw', 'ready-to-use',
    'refrigerated', 'ripe', 'room', 'skin-on', 'skinless', 'soft', 'stiff',
    'temperature', 'uncooked', 'undrained', 'unripe', 'warm', 'washed', 'whole',

    # --- Size & Shape ---
    'bite-sized', 'chunky', 'clump', 'coarse', 'fine', 'jumbo', 'large', 'long',
    'medium', 'round', 'short', 'small', 'thick', 'thin',

    # --- Quantifiers & Qualifiers ---
    'about', 'additional', 'approximately', 'bunch', 'coarse', 'extra', 'generous',
    'heavy', 'heaping', 'level', 'more', 'packed', 'plus', 'scant', 'splash',
    'sprig', 'sprinkle',

    # --- Flavor & Taste ---
    'bitter', 'salty', 'savory', 'sour', 'spicy', 'sweet', 'sweetened', 'unsalted',
    'unsweetened',

    # --- Common Stop Words (Articles, Conjunctions, Prepositions) ---
    'a', 'an', 'and', 'as', 'at', 'for', 'in', 'into', 'of', 'on', 'or', 'the',
    'to', 'with', 'without',

    # --- Instructions & Meta-words ---
    'divided', 'dusting', 'garnish', 'needed', 'serving', 'taste',
}

def parse_ingredient(ingredient_string: str) -> str:
    """
    Parses a raw ingredient string to extract its essential name.

    This function cleans the input string by performing a series of sequential
    operations:
    1.  Converts the string to lowercase.
    2.  Removes text within parentheses (e.g., "(optional)").
    3.  Removes numerical quantities, including fractions and decimals.
    4.  Removes punctuation.
    5.  Splits the string into words and removes common units and descriptors.
    6.  Reassembles the string and normalizes whitespace.

    Args:
        ingredient_string: The raw ingredient string from a recipe.
                           Example: "2 1/2 cups (12.5 oz) sifted all-purpose flour, for dusting"

    Returns:
        A cleaned, normalized string representing the core ingredient.
        Example: "all-purpose flour"
    """
    if not isinstance(ingredient_string, str) or not ingredient_string:
        return ""

    # 1. Convert to lowercase for consistent processing.
    text = ingredient_string.lower()

    # 2. Remove parenthetical remarks (e.g., "(optional)", "(about 1 pound)").
    text = re.sub(r'\([^)]*\)', '', text)

    # 3. Remove numerical quantities, including integers, decimals, and fractions.
    # This regex handles formats like "1 1/2", "1/2", "1.5", "1".
    text = re.sub(r'(\d+\s+)?\d+/\d+|\d+(\.\d+)?|\d+', '', text)

    # 4. Remove common punctuation. We keep hyphens as they can be part of a name.
    text = re.sub(r'[,.;:?!"]', '', text)

    # 5. Tokenize and filter out units and descriptors.
    words = text.split()
    # This list comprehension is efficient for filtering. We check against the
    # predefined sets of UNITS and DESCRIPTORS.
    clean_words = [
        word for word in words if word not in UNITS and word not in DESCRIPTORS
    ]

    # 6. Reassemble the string and clean up whitespace.
    # ' '.join() handles the list-to-string conversion.
    # The final split/join is a robust way to normalize multiple spaces to single spaces.
    clean_name = ' '.join(clean_words).strip()

    return clean_name

# Make sure you have `transformers` and a backend like `torch` or `tensorflow` installed.
# pip install transformers torch
try:
    from transformers import pipeline, Pipeline
except ImportError:
    print("Warning: `transformers` library not found. ML-based classification will not be available.")
    print("Please run 'pip install transformers torch' to install.")
    pipeline = None
    Pipeline = None

# Import the parser from the previous step.
# Assume it's in a file named `ingredient_parser.py` in the same directory.


# ==============================================================================
#  Global State and Constants for Classifier
# ==============================================================================

# Caching mechanism to store results and avoid re-computation, critical for performance.
# Key: clean ingredient name (str), Value: vegan status (bool)
VEGAN_CACHE: Dict[str, bool] = {}

# Lazy-loaded Hugging Face pipeline. Initialized only when first needed.
CLASSIFIER_PIPELINE: Optional[Pipeline] = None

# --- Rule-Based Keyword Sets ---

# Keywords for ingredients that are definitively NOT vegan.
# This list is comprehensive to catch common animal products quickly.
NON_VEGAN_KEYWORDS: Set[str] = {
    # --- Meats (Red & White) ---
    'andouille', 'bacon', 'beef', 'biltong', 'bison', 'boar', 'bologna', 'bratwurst', 'brisket', 'capicola', 'chorizo', 'chops', 'corned beef', 'frankfurter', 'goat', 'ground chuck', 'guanciale', 'ham', 'head cheese', 'jerky', 'kebab', 'kielbasa', 'kidney', 'lamb', 'liver', 'meat', 'meatball', 'meatballs', 'mince', 'mortadella', 'mutton', 'pancetta', 'pastrami', 'pemmican', 'pepperoni', 'pork', 'prosciutto', 'ribs', 'salami', 'sausage', 'shank', 'soppressata', 'steak', 'sweetbreads', 'tenderloin', 'tongue', 'tripe', 'veal', 'venison',
    # --- Poultry ---
    'albumen', 'capon', 'chicken', 'confit', 'cornish hen', 'duck', 'egg', 'eggs', 'foie gras', 'giblets', 'goose', 'guinea fowl', 'meringue', 'nuggets', 'ostrich', 'partridge', 'pate', 'pheasant', 'poultry', 'quail', 'turkey', 'yolk',
    # --- Seafood (Fish) ---
    'anchovies', 'bass', 'bluefish', 'carp', 'catfish', 'caviar', 'cod', 'eel', 'escargot', 'fish', 'flounder', 'gefilte fish', 'grouper', 'haddock', 'halibut', 'herring', 'lox', 'mahi-mahi', 'mackerel', 'monkfish', 'perch', 'pickerel', 'pollock', 'roe', 'salmon', 'sardine', 'seabass', 'sole', 'sturgeon', 'surimi', 'swordfish', 'tilapia', 'trout', 'tuna', 'walleye',
    # --- Seafood (Shellfish & Other) ---
    'abalone', 'calamari', 'clams', 'cockle', 'conch', 'crab', 'crawfish', 'crayfish', 'cuttlefish', 'krill', 'langostino', 'lobster', 'mussels', 'octopus', 'oyster', 'oysters', 'prawns', 'scallop', 'scallops', 'scampi', 'sea urchin', 'seafood', 'shrimp', 'squid', 'uni', 'whelk',
    # --- Dairy ---
    'asiago', 'bleu', 'brie', 'butter', 'buttermilk', 'camembert', 'casein', 'caseinate', 'cheddar', 'cheese', 'colby', 'cottage', 'cream', 'creme', 'curd', 'edam', 'feta', 'ghee', 'gorgonzola', 'gouda', 'gruyere', 'half-and-half', 'halloumi', 'havarti', 'kefir', 'lactalbumin', 'lactose', 'manchego', 'mascarpone', 'milk', 'monterey jack', 'mozzarella', 'muenster', 'neufchatel', 'paneer', 'parmesan', 'provolone', 'queso', 'ricotta', 'sour cream', 'whey', 'yogurt',
    # --- Animal Fats, By-products & Additives ---
    'ambergris', 'aspic', 'bone char', 'bone meal', 'bone marrow', 'bouillon', 'broth', 'carmine', 'chitin', 'cochineal', 'collagen', 'consomme', 'demi-glace', 'drippings', 'fat', 'fish oil', 'gelatin', 'glycerides', 'glycerol', 'isinglass', 'keratin', 'l-cysteine', 'lanolin', 'lard', 'lipase', 'musk', 'pepsin', 'rennet', 'schmaltz', 'shellac', 'stearic acid', 'stock', 'suet', 'tallow', 'vitamin d3',
    # --- Bee Products ---
    'bee pollen', 'beeswax', 'honey', 'propolis', 'royal jelly',
}

# --- ALWAYS_VEGAN_KEYWORDS ---
# A comprehensive list of common ingredients that are almost always vegan. 
# This helps quickly classify simple ingredients without needing the ML model.
ALWAYS_VEGAN_KEYWORDS: Set[str] = {
    # --- Staples & Dry Goods ---
    'arrowroot', 'baking powder', 'baking soda', 'beans', 'bread crumbs', 'chickpeas', 'cornmeal', 'cornstarch', 'couscous', 'flour', 'lentils', 'pasta', 'quinoa', 'rice', 'sugar', 'yeast',
    # --- Fats & Oils ---
    'margarine', 'oil', 'canola oil', 'coconut oil', 'olive oil', 'sesame oil', 'sunflower oil', 'vegetable oil',
    # --- Seasonings, Herbs & Spices ---
    'basil', 'black pepper', 'cayenne', 'chili', 'cilantro', 'cinnamon', 'clove', 'cumin', 'curry', 'garlic', 'ginger', 'herbs', 'mustard', 'nutmeg', 'onion', 'oregano', 'paprika', 'parsley', 'pepper', 'rosemary', 'saffron', 'salt', 'spices', 'thyme', 'turmeric',
    # --- Liquids & Acids ---
    'coffee', 'club soda', 'lemon juice', 'lime juice', 'seltzer', 'soda', 'tea', 'vegetable stock', 'vegetable broth', 'vinegar', 'water',
}


# --- VEGAN_ALTERNATIVE_PREFIXES ---
# A list of prefixes for plant-based alternatives that can otherwise be misidentified
# by simple keyword matching (e.g., 'soy milk', 'cashew cheese').
VEGAN_ALTERNATIVE_PREFIXES: Set[str] = {
    # --- Nuts & Seeds ---
    'almond', 'cashew', 'flax', 'hazelnut', 'hemp', 'macadamia', 'pecan', 'pistachio', 'pumpkinseed', 'sesame', 'sunflower', 'walnut',
    # --- Grains & Legumes ---
    'chickpea', 'lentil', 'oat', 'pea', 'quinoa', 'rice', 'soy',
    # --- Fruits & Vegetables ---
    'apple', 'avocado', 'banana', 'potato', 'vegetable', 'veggie',
    # --- Other ---
    'cocoa', 'coconut', 'plant', 'plant-based', 'vegan',
}

ALWAYS_KETO_KEYWORDS = {
    'splenda', 'stevia', 'erythritol', 'monkfruit', 'sucralose', 'allulose'
}
def _get_classifier() -> Optional[Pipeline]:
    """
    Initializes and returns the Hugging Face text classification pipeline.

    This function uses lazy loading: the model is only loaded into memory
    the first time it's needed, preventing slow startup times.
    It is robust against the absence of the `transformers` library.
    """
    global CLASSIFIER_PIPELINE
    if CLASSIFIER_PIPELINE is None:
        if pipeline is None:
            # transformers library is not installed
            return None
        try:
            print("INFO: Initializing Hugging Face classifier for the first time. This may take a moment...")
            # Using the model mentioned in the task prompt.
            CLASSIFIER_PIPELINE  = pipeline("text-classification", model="nisuga/food_type_classification_model")
            print("INFO: Classifier initialized successfully.")
        except Exception as e:
            print(f"ERROR: Failed to load Hugging Face model '{model_name}'. ML classification will be disabled.")
            print(f"Error details: {e}")
            # Set to a dummy value to prevent re-trying
            CLASSIFIER_PIPELINE = "failed"
    
    if CLASSIFIER_PIPELINE == "failed":
        return None
        
    return CLASSIFIER_PIPELINE


def is_ingredient_vegan(ingredient_string: str) -> bool:
    """
    Classifies a single ingredient as vegan or not using a multi-layered approach.

    The process is as follows:
    1. Parse the ingredient to get a clean name.
    2. Check a cache for a previously computed result.
    3. Apply a series of precise rules to handle common cases and pitfalls
       (e.g., "eggless", "peanut butter", "soy milk").
    4. If no rule applies, use a Hugging Face ML model for classification.
    5. Cache the final result before returning.

    Args:
        ingredient_string: The raw ingredient string from a recipe.

    Returns:
        True if the ingredient is determined to be vegan, False otherwise.
    """

    # 1. Parse the ingredient string to get a clean, standardized name.
    
    clean_name = parse_ingredient(ingredient_string)

    if not clean_name:
        return True  # An empty or unparsable ingredient is assumed to not make a dish non-vegan.

    # 2. Check cache for a quick result.
    if clean_name in VEGAN_CACHE:
        
        return VEGAN_CACHE[clean_name]

    # --- 3. Apply Rule-Based Logic (Fast and Accurate Checks) ---
    
    # Pitfall: Handle "eggless" before checking for "egg".
    if "eggless" in clean_name:
        VEGAN_CACHE[clean_name] = True
        return True

    # Pitfall: Handle plant-based butters/milks before general "butter"/"milk" checks.
    if any(prefix in clean_name for prefix in VEGAN_ALTERNATIVE_PREFIXES):
        # e.g., "peanut butter", "soy milk"
        VEGAN_CACHE[clean_name] = True
        return True

    # General check for common non-vegan keywords.
    # We split the clean name to avoid false positives (e.g., "ham" in "shame").
    words_in_name = set(clean_name.split())
    if not NON_VEGAN_KEYWORDS.isdisjoint(words_in_name):
        VEGAN_CACHE[clean_name] = False
        
        return False

    # Check for always-vegan ingredients.
    if clean_name in ALWAYS_VEGAN_KEYWORDS:
        VEGAN_CACHE[clean_name] = True
        return True

    # --- 4. Fallback to ML Model for Ambiguous Cases ---
    
    # Assume non-vegan as a safe default if ML model fails or is unavailable.
    result = False
    
    classifier = _get_classifier()
    if classifier:
        try:
            prediction = classifier(clean_name, top_k=1)[0]
            
            # The model labels are 'plant-based' and 'animal-based'.
            if prediction['label'] == 'PLANT_BASED':
                result = True
        except Exception as e:
            print(f"WARNING: ML classification failed for '{clean_name}'. Defaulting to non-vegan. Error: {e}")
            result = False
    else:
        print(f"WARNING: No ML classifier available. Defaulting '{clean_name}' to non-vegan.")
        result = False

    # 5. Cache the result before returning.
    VEGAN_CACHE[clean_name] = result
    return result



# --- Configuration ---
# Path to the directory where you unzipped the SR Legacy files.
DATA_SOURCE_PATH = './sr_legacy/'

# The name for our final, clean database file.
OUTPUT_DB_PATH = './data/nutrition_database.csv'

# Column names for the legacy text files, based on USDA documentation.
FOOD_DES_COLS = [
    'NDB_No', 'FdGrp_Cd', 'Long_Desc', 'Shrt_Desc', 'ComName', 'ManufacName', 
    'Survey', 'Ref_desc', 'Refuse', 'SciName', 'N_Factor', 'Pro_Factor', 
    'Fat_Factor', 'CHO_Factor'
]

NUT_DATA_COLS = [
    'NDB_No', 'Nutr_No', 'Nutr_Val', 'Num_Data_Pts', 'Std_Error', 'Src_Cd', 
    'Deriv_Cd', 'Ref_NDB_No', 'Add_Nutr_Mark', 'Num_Studies', 'Min', 'Max', 
    'DF', 'Low_EB', 'Up_EB', 'Stat_cmt', 'AddMod_Date', 'CC'
]

NUTR_DEF_COLS = [
    'Nutr_No', 'Units', 'Tagname', 'NutrDesc', 'Num_Dec', 'SR_Order'
]

# The specific nutrients we want to extract.
# `NutrDesc` is the column name in the legacy format.
TARGET_NUTRIENTS = {
    'Carbohydrate, by difference': 'carbs',
    'Protein': 'protein',
    'Sugars, total': 'sugar' # Note: In some versions, it's 'Sugars, total including NLEA'
}

def create_nutrition_database_from_txt():
    """
    Loads the raw USDA SR Legacy TXT files, processes them, and saves a clean,
    wide-format nutritional database as a single CSV file.
    
    This version is specifically adapted to parse the tilde/caret delimited format.
    """
    print("--- Starting Nutrition Database Preparation (from TXT files) ---")

    # --- 1. Load the necessary TXT files with correct parsing ---
    try:
        print(f"Loading data from '{DATA_SOURCE_PATH}'...")
        # food.csv equivalent is FOOD_DES.txt
        food_df = pd.read_csv(
            os.path.join(DATA_SOURCE_PATH, 'FOOD_DES.txt'),
            sep='^',
            quotechar='~',
            header=None,
            names=FOOD_DES_COLS,
            encoding='latin1' # This encoding is often needed for these files
        )

        # nutrient.csv equivalent is NUTR_DEF.txt
        nutrient_df = pd.read_csv(
            os.path.join(DATA_SOURCE_PATH, 'NUTR_DEF.txt'),
            sep='^',
            quotechar='~',
            header=None,
            names=NUTR_DEF_COLS,
            encoding='latin1'
        )

        # food_nutrient.csv equivalent is NUT_DATA.txt
        food_nutrient_df = pd.read_csv(
            os.path.join(DATA_SOURCE_PATH, 'NUT_DATA.txt'),
            sep='^',
            quotechar='~',
            header=None,
            names=NUT_DATA_COLS,
            encoding='latin1'
        )
        
    except FileNotFoundError as e:
        print(f"ERROR: Could not find required TXT files in '{DATA_SOURCE_PATH}'.")
        print("Please ensure FOOD_DES.txt, NUTR_DEF.txt, and NUT_DATA.txt are present.")
        print(f"Details: {e}")
        return

    # --- 2. Filter for only the nutrients we care about ---
    #print(f"Filtering for target nutrients: {list(TARGET_NUTRIENTS.keys())}")
    target_nutrient_ids = nutrient_df[nutrient_df['NutrDesc'].isin(TARGET_NUTRIENTS.keys())]
    
    filtered_food_nutrient_df = food_nutrient_df[
        food_nutrient_df['Nutr_No'].isin(target_nutrient_ids['Nutr_No'])
    ]

    # --- 3. Merge the tables to get food and nutrient names ---
    #print("Merging data tables...")
    merged_df = pd.merge(
        filtered_food_nutrient_df,
        food_df[['NDB_No', 'Long_Desc']],
        on='NDB_No',
        how='left'
    )
    merged_df = pd.merge(
        merged_df,
        nutrient_df[['Nutr_No', 'NutrDesc']],
        on='Nutr_No',
        how='left'
    )

    # --- 4. Pivot the table from "long" to "wide" format ---
    #print("Pivoting table to wide format...")
    # Use the original legacy column names for pivoting
    nutrition_pivot_df = merged_df.pivot_table(
        index='Long_Desc',
        columns='NutrDesc',
        values='Nutr_Val'
    ).reset_index()

    # --- 5. Clean up the final DataFrame ---
    #print("Cleaning up the final database...")
    nutrition_pivot_df = nutrition_pivot_df.rename(columns=TARGET_NUTRIENTS)
    nutrition_pivot_df = nutrition_pivot_df.rename(columns={'Long_Desc': 'food_name'})
    
    cols_to_fill = ['carbs', 'protein', 'sugar']
    for col in cols_to_fill:
        if col not in nutrition_pivot_df.columns:
            nutrition_pivot_df[col] = 0.0
    nutrition_pivot_df[cols_to_fill] = nutrition_pivot_df[cols_to_fill].fillna(0.0)
    
    nutrition_pivot_df['food_name'] = nutrition_pivot_df['food_name'].str.lower()
    
    # --- 6. Save the final database ---
    os.makedirs(os.path.dirname(OUTPUT_DB_PATH), exist_ok=True)
    nutrition_pivot_df.to_csv(OUTPUT_DB_PATH, index=False)
    
create_nutrition_database_from_txt()

# ==============================================================================
# VEGAN CLASSIFIER
# ==============================================================================

    
def is_vegan(ingredients):
    for ingredient in ingredients:
        clean_ingredient = parse_ingredient(ingredient).lower()
        ingredient_words = set(clean_ingredient.split())
        if not NON_VEGAN_KEYWORDS.isdisjoint(ingredient_words):
            return False
    return all(map(is_ingredient_vegan, ingredients))
    


# ==============================================================================
#  Keto
# ==============================================================================

try:
    from sklearn.metrics import classification_report, confusion_matrix
except ImportError:
    def classification_report(y_true, y_pred, **kwargs):
        print("Warning: scikit-learn not found. Skipping classification report.", file=sys.stderr)
        return "scikit-learn not installed."
    
    def confusion_matrix(y_true, y_pred):
        print("Warning: scikit-learn not found. Cannot generate confusion matrix.", file=sys.stderr)
        return [[0, 0], [0, 0]]


UNITS: Set[str] = {"c", "cup", "cups", "g", "gram", "grams", "kg", "kilogram", "kilograms", "lb", "lbs", "pound", "pounds", "ml", "milliliter", "milliliters", "oz", "ounce", "ounces", "pinch", "pinches", "splash", "splashes", "sprig", "sprigs", "t", "tsp", "teaspoon", "teaspoons", "T", "tbsp", "tablespoon", "tablespoons", "can", "cans", "clove", "cloves", "dash", "dashes", "drizzle", "drop", "drops", "gallon", "gallons", "handful", "handfuls", "head", "heads", "package", "packages", "packet", "packets", "pint", "pints", "quart", "quarts", "scoop", "scoops", "sheet", "sheets", "slice", "slices", "stalk", "stalks", "stick", "sticks", "strip", "strips"}

DESCRIPTORS: Set[str] = {'beaten', 'blanched', 'boiled', 'braised', 'brewed', 'brined', 'broken', 'charred', 'chilled', 'chopped', 'clarified', 'coarsely', 'crumbled', 'crushed', 'cubed', 'cut', 'deboned', 'deglazed', 'deseeded', 'deveined', 'diced', 'dissolved', 'divided', 'drained', 'finely', 'flaked', 'folded', 'grated', 'grilled', 'halved', 'heated', 'hulled', 'husked', 'infused', 'julienned', 'juiced', 'kneaded', 'marinated', 'mashed', 'melted', 'minced', 'mixed', 'parboiled', 'patted', 'peeled', 'pitted', 'poached', 'pounded', 'prepared', 'pressed', 'pureed', 'quartered', 'rinsed', 'roasted', 'rolled', 'roughly', 'scalded', 'scored', 'scrubbed', 'seared', 'seeded', 'segmented', 'shaved', 'shredded', 'shucked', 'sifted', 'skewered', 'sliced', 'slivered', 'smashed', 'soaked', 'softened', 'squeezed', 'steamed', 'stemmed', 'stewed', 'strained', 'stuffed', 'thawed', 'thinly', 'tied', 'toasted', 'torn', 'trimmed', 'whisked', 'zested', 'canned', 'cold', 'condensed', 'cooked', 'cooled', 'cored', 'creamed', 'cured', 'defrosted', 'fermented', 'firmly', 'freshly', 'frozen', 'hard', 'hot', 'instant', 'jarred', 'lean', 'leftover', 'light', 'lukewarm', 'optional', 'pasteurized', 'powdered', 'preserved', 'raw', 'ready-to-use', 'refrigerated', 'ripe', 'room', 'skin-on', 'skinless', 'soft', 'stiff', 'temperature', 'uncooked', 'undrained', 'unripe', 'warm', 'washed', 'whole', 'bite-sized', 'chunky', 'clump', 'coarse', 'fine', 'jumbo', 'large', 'long', 'medium', 'round', 'short', 'small', 'thick', 'thin', 'about', 'additional', 'approximately', 'bunch', 'extra', 'generous', 'heavy', 'heaping', 'level', 'more', 'packed', 'plus', 'scant', 'splash', 'sprig', 'sprinkle', 'bitter', 'salty', 'savory', 'sour', 'spicy', 'sweetened', 'unsalted', 'unsweetened', 'a', 'an', 'and', 'as', 'at', 'for', 'in', 'into', 'of', 'on', 'or', 'the', 'to', 'with', 'without', 'dusting', 'garnish', 'needed', 'serving', 'taste'}

def _depluralize(word: str) -> str:
    """A more robust de-pluralizer using the inflect library."""
    # Fallback to simple version if inflect is not installed
    if word.endswith('ss'): return word
    if word.endswith('s'): return word[:-1]
    return word

def parse_ingredient(ingredient_string: str) -> str:
    """An improved parser that cleans an ingredient string and de-pluralizes it."""
    if not isinstance(ingredient_string, str) or not ingredient_string: return ""
    text = ingredient_string.lower()
    text = re.sub(r'\([^)]*\)', '', text)
    text = re.sub(r'(\d+\s+)?\d+/\d+|\d+(\.\d+)?|\d+', '', text)
    text = re.sub(r'[,.;:?!"]', '', text)
    words = text.split()
    clean_words = [_depluralize(word) for word in words if word not in UNITS and word not in DESCRIPTORS]
    return ' '.join(clean_words).strip()

# ==============================================================================
# SECTION 2: KETO CLASSIFIER
# ==============================================================================

# --- Tuning Parameters & Knowledge Bases ---
CONFIDENCE_THRESHOLD = 80
LOG_FAILED_LOOKUPS = False


# FIXED: Updated keyword lists with better coverage
MANUAL_KETO_OVERRIDES = {
    'water', 'salt', 'oil', 'butter', 'avocado', 'egg', 'chicken', 'beef', 'pork', 
    'lamb', 'fish', 'salmon', 'tuna', 'shrimp', 'cheese', 'mayonnaise', 'vinegar', 
    'mustard', 'splenda', 'stevia', 'erythritol', 'garlic', 'paprika', 'olive',
    'cream', 'bacon', 'lettuce', 'spinach', 'broccoli', 'cauliflower', 'pepper',
    'onion', 'mushroom', 'lemon', 'lime', 'herbs', 'spice'
}

NON_KETO_KEYWORDS = { 
    # --- Grains, Flours & Starches ---
    'barley', 'bread', 'breadcrumb', 'cereal', 'corn', 'cornmeal', 'cornstarch', 'couscous', 
    'cracker', 'crouton', 'flour', 'granola', 'millet', 'oat', 'pasta', 'panko',
    'pretzel', 'quinoa', 'rice', 'rye', 'semolina', 'spelt', 'tapioca', 'tortilla', 'wheat',
    'baguette', 'bannock', 'ciabatta', 'noodle', 'rusk',

    # --- Starchy Vegetables & Tubers ---
    'parsnip', 'pea', 'plantain', 'potato', 'yam',

    # --- High-Sugar Fruits ---
    'banana', 'cherry', 'date', 'fig', 'grape', 'lychee', 'mango', 'pineapple', 
    'tangerine', 'apple', 'orange', 'strawberry', 'strawberrie', 
    
    # --- Alcoholic Beverages (High Carb) ---
    'vodka', 'rum', 'liqueur', 'beer', 'wine',  # FIXED: Added alcohol
    
    # --- Sugars & Syrups ---
    'agave', 'caramel', 'dextrose', 'fructose', 'glucose', 'honey', 
    'maltodextrin', 'maple', 'molasses', 'sugar', 'syrup',

    # --- Legumes ---
    'bean', 'chickpea', 'lentil', 'legume',

    # --- Processed Foods & Desserts ---
    'biscuit', 'cake', 'candy', 'chip', 'chocolate', 'cookie', 'dough',
    'doughnut', 'dumpling', 'ice cream', 'jam', 'jelly', 'ketchup', 'muffin', 'pie', 
    'pizza', 'popcorn', 'relish', 'sorbet', 'waffle',
}

# Database & Fuzzy Matching
DB_PATH = './data/nutrition_database.csv'
NUTRITION_DF: Optional[pd.DataFrame] = None
FOOD_NAME_CHOICES: Optional[List[str]] = None
SEARCH_CACHE: Dict[str, Optional[Dict]] = {}
KETO_CARB_THRESHOLD_PER_100G = 10.0

def _load_database() -> None:
    global NUTRITION_DF, FOOD_NAME_CHOICES
    if NUTRITION_DF is not None: return
    try:
        NUTRITION_DF = pd.read_csv(DB_PATH)
        FOOD_NAME_CHOICES = NUTRITION_DF['food_name'].tolist()
    except FileNotFoundError:
        print(f"CRITICAL ERROR: Nutritional database not found at '{DB_PATH}'. Aborting.", file=sys.stderr)
        sys.exit(1)

def find_ingredient_nutrition(ingredient_name: str) -> Optional[Dict[str, Any]]:
    """
    Searches for an ingredient in the nutritional database.
    Returns high-carb dummy data for failed lookups to ensure conservative classification.
    """
    _load_database()
    if NUTRITION_DF is None or NUTRITION_DF.empty:
        return {'food_name': f"db_load_fail: {ingredient_name}", 'carbs': 1001.0, 'protein': 0, 'sugar': 0}
        
    if ingredient_name in SEARCH_CACHE:
        return SEARCH_CACHE[ingredient_name]
        
    query_words = set(ingredient_name.split())
    if not query_words:
        result = {'food_name': f"empty_query: {ingredient_name}", 'carbs': 1001.0, 'protein': 0, 'sugar': 0}
        SEARCH_CACHE[ingredient_name] = result
        return result

    # Pre-filter choices
    filtered_choices = [name for name in FOOD_NAME_CHOICES if not query_words.isdisjoint(name.lower().split())]
    if not filtered_choices:
        if LOG_FAILED_LOOKUPS:
            with open("failed_lookups.log", "a") as f: 
                f.write(f"PREFILTER_FAIL: {ingredient_name}\n")
        result = {'food_name': f"prefilter_fail: {ingredient_name}", 'carbs': 1001.0, 'protein': 0, 'sugar': 0}
        SEARCH_CACHE[ingredient_name] = result
        return result

    # Fuzzy matching
    best_match = process.extractOne(ingredient_name, filtered_choices, scorer=fuzz.WRatio)
    
    if best_match and best_match[1] >= CONFIDENCE_THRESHOLD:
        matched_food_name = best_match[0]
        result = NUTRITION_DF[NUTRITION_DF['food_name'] == matched_food_name].iloc[0].to_dict()
    else:
        # FIXED: Failed matches get HIGH carbs (not 0.0)
        if LOG_FAILED_LOOKUPS:
            with open("failed_lookups.log", "a") as f: 
                f.write(f"SCORE_FAIL: {ingredient_name}\n")
        result = {'food_name': f"score_fail: {ingredient_name}", 'carbs': 1001.0, 'protein': 0, 'sugar': 0}

    SEARCH_CACHE[ingredient_name] = result
    return result

def is_ingredient_keto(ingredient: str, debug: bool = False) -> bool:
    """
    Enhanced keto checker with debug output and better logic.
    """
    clean_name = parse_ingredient(ingredient)
    
    if debug:
        print(f"  Checking: '{ingredient[:50]}...' -> cleaned: '{clean_name}'")
    
    # Empty ingredients are safe
    if not clean_name:
        if debug: print(f"    ✓ Empty ingredient - KETO")
        return True

    # Fast fail for known non-keto ingredients (CHECK THIS FIRST!)
    ingredient_words = set(clean_name.split())
    if not NON_KETO_KEYWORDS.isdisjoint(ingredient_words):
        matching_words = NON_KETO_KEYWORDS.intersection(ingredient_words)
        if debug: print(f"    ✗ Contains non-keto keyword(s): {matching_words} - NOT KETO")
        return False

    # Fast pass for known keto ingredients
    if not MANUAL_KETO_OVERRIDES.isdisjoint(ingredient_words):
        matching_words = MANUAL_KETO_OVERRIDES.intersection(ingredient_words)
        if debug: print(f"    ✓ Contains keto override(s): {matching_words} - KETO")
        return True
        
    # Database lookup for ambiguous ingredients
    nutrition_data = find_ingredient_nutrition(clean_name)
    if nutrition_data:
        carbs = nutrition_data.get('carbs', KETO_CARB_THRESHOLD_PER_100G + 1)
        is_keto_by_carbs = carbs <= KETO_CARB_THRESHOLD_PER_100G
        if debug:
            print(f"    Database: {nutrition_data['food_name']}, carbs: {carbs}g -> {'KETO' if is_keto_by_carbs else 'NOT KETO'}")
        return is_keto_by_carbs

    # Default to NOT KETO for unknown ingredients
    if debug: print(f"    ? Unknown ingredient, defaulting to NOT KETO")
    return False

def is_keto(ingredients: List[str], debug: bool = False) -> bool:
    """
    Determines if a recipe is keto with optional debug output.
    """
    if not isinstance(ingredients, list):
        return False
        
    if debug:
        print(f"\n--- Analyzing recipe with {len(ingredients)} ingredients ---")
        
    for ingredient_str in ingredients:
        if not is_ingredient_keto(ingredient_str, debug=debug):
            if debug:
                print(f"  --> Recipe Verdict: NOT KETO (Failed on: '{ingredient_str}')")
            return False
            
    if debug:
        print(f"  --> Recipe Verdict: KETO (All ingredients passed)")
    return True


# ==============================================================================
# MAIN EXECUTION & REPORTING
# ==============================================================================

def parse_malformed_ingredient_string(s: str) -> List[str]:
    """Parses a string that looks like a list but may be missing commas."""
    if not isinstance(s, str): 
        return []
    
    # First try to parse as a literal list
    try: 
        result = ast.literal_eval(s)
        if isinstance(result, list):
            return result
    except (ValueError, SyntaxError):
        pass
    
    # Try to extract quoted strings
    quoted_matches = re.findall(r"'([^']*)'", s)
    if quoted_matches:
        return quoted_matches
    
    # If it's a single long string that looks like concatenated ingredients,
    # try to split it intelligently
    if len(s) > 100 and not s.startswith('['):  # Likely a concatenated string
        # Split on common ingredient separators and numbers+units
        # This is a heuristic approach for badly formatted data
        ingredients = []
        
        # Split on patterns like "2 cups", "1 tablespoon", etc.
        parts = re.split(r'(?=\d+(?:\s+\d+/\d+)?\s+(?:cup|tablespoon|teaspoon|pound|ounce|clove|package))', s)
        
        for part in parts:
            if part.strip():
                ingredients.append(part.strip())
        
        if len(ingredients) > 1:
            print(f"DEBUG: Split concatenated string into {len(ingredients)} parts")
            return ingredients
    
    # Fallback: treat as single ingredient
    return [s] if s.strip() else []

def print_confusion_matrix(y_true, y_pred, classifier_type: str):
    """
    Calculates and prints a labeled confusion matrix for a given classifier type.

    Args:
        y_true: The true labels from the ground truth data.
        y_pred: The predicted labels from the classifier.
        classifier_type: A string, either 'keto' or 'vegan', to set the labels.
    """
    # First, ensure scikit-learn was imported successfully
    if 'confusion_matrix' not in globals():
        print("Warning: scikit-learn not found. Skipping confusion matrix.", file=sys.stderr)
        return

    try:
        # --- 1. Define labels based on the classifier type ---
        if classifier_type.lower() == 'keto':
            positive_class = 'Keto'
            negative_class = 'Non-Keto'
        elif classifier_type.lower() == 'vegan':
            positive_class = 'Vegan'
            negative_class = 'Non-Vegan'
        else:
            # A fallback for any other case
            positive_class = 'Positive'
            negative_class = 'Negative'

        # --- 2. Dynamically generate the labels for the DataFrame ---
        index_labels = [f'Actual: {negative_class}', f'Actual: {positive_class}']
        column_labels = [f'Predicted: {negative_class}', f'Predicted: {positive_class}']

        cm = confusion_matrix(y_true, y_pred)
        cm_df = pd.DataFrame(cm, index=index_labels, columns=column_labels)

        print(f"\n--- Confusion Matrix: {classifier_type.capitalize()} Classifier ---")
        print(cm_df)
        print("-" * 55)
        
        # Unpack the matrix values
        tn, fp, fn, tp = cm.ravel()
        
        # --- 3. Dynamically generate the descriptive printout ---
        print(f"  True Negatives (TN): {tn: >4} (Correctly predicted {negative_class})")
        print(f"  False Positives(FP): {fp: >4} (Incorrectly predicted {positive_class})")
        print(f"  False Negatives(FN): {fn: >4} (Incorrectly predicted {negative_class})")
        print(f"  True Positives (TP): {tp: >4} (Correctly predicted {positive_class})")
        print("-" * 55)
        
    except Exception as e:
        print(f"Could not generate confusion matrix for '{classifier_type}': {e}")

def main(args):
    """Main function to load data, run classifiers, and print a full report."""
    ground_truth = pd.read_csv(args.ground_truth, index_col=None)
    ground_truth['ingredients'] = ground_truth['ingredients'].apply(parse_malformed_ingredient_string)
    

    _load_database()
    start_time = time()
    ground_truth['keto_pred'] = ground_truth['ingredients'].apply(lambda x: is_keto(x, debug=False))
    ground_truth['vegan_pred'] = ground_truth['ingredients'].apply(is_vegan)
    end_time = time()


    # --- UPDATED REPORTING SECTION ---
    print("\n" + "="*20 + " FINAL CLASSIFICATION REPORT " + "="*20)

    # --- Keto Report ---
    print("\n" + "="*20 + " Keto " + "="*20)
    print(classification_report(ground_truth['keto'], ground_truth['keto_pred']))
    # Call the new function with classifier_type='keto'
    print_confusion_matrix(ground_truth['keto'], ground_truth['keto_pred'], classifier_type='keto')

    # --- Vegan Report ---
    print("\n" + "="*20 + " Vegan " + "="*20)
    print(classification_report(ground_truth['vegan'], ground_truth['vegan_pred']))
    # Call the new function with classifier_type='vegan'
    print_confusion_matrix(ground_truth['vegan'], ground_truth['vegan_pred'], classifier_type='vegan')

    print(f"\n== Time taken: {end_time:.2f} seconds ==")
    
    return 0

if __name__ == "__main__":
    parser = ArgumentParser(description="Run diet classifiers on a ground truth dataset.")
    parser.add_argument("--ground_truth", type=str, default="/usr/src/data/ground_truth_sample.csv",
                        help="Path to the ground truth CSV file.")
    sys.exit(main(parser.parse_args()))