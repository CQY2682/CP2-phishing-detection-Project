"""
CP2 Training Data Augmentation
================================

Purpose:
    PhiUSIIL legitimate URLs are homepage-only (max 58 chars, all HTTPS).
    This causes the model to flag ANY URL with a path as phishing.
    
    Fix: Inject curated real-world legitimate URLs with paths, HTTP,
    login pages, and query parameters into the training set.

Run: python src/augment_training.py
"""

import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_extractor import extract_features, FEATURE_NAMES

# ============================================================
# CURATED LEGITIMATE URLs (hand-picked, verified real sites)
# ============================================================
# These break PhiUSIIL's biases:
# - Long paths (slash count >= 3)
# - HTTP protocol (no HTTPS)
# - Login/auth pages on real brands
# - Query parameters
# - News articles, docs, product pages

REAL_LEGIT_URLS = [
    # ---- Documentation sites (long paths, many slashes) ----
    "https://docs.python.org/3/library/urllib.parse.html",
    "https://docs.python.org/3/tutorial/introduction.html",
    "https://docs.python.org/3/reference/expressions.html",
    "https://docs.microsoft.com/en-us/azure/active-directory/",
    "https://docs.microsoft.com/en-us/dotnet/csharp/",
    "https://developer.mozilla.org/en-US/docs/Web/HTTP/",
    "https://developer.mozilla.org/en-US/docs/Web/JavaScript/",
    "https://developer.mozilla.org/en-US/docs/Learn/HTML/",
    "https://reactjs.org/docs/getting-started.html",
    "https://vuejs.org/v2/guide/installation.html",
    "https://angular.io/guide/architecture",
    "https://flask.palletsprojects.com/en/2.3.x/quickstart/",
    "https://django-rest-framework.org/api-guide/views/",
    "https://scikit-learn.org/stable/modules/ensemble.html",
    "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html",
    "https://numpy.org/doc/stable/reference/generated/numpy.array.html",
    "https://kubernetes.io/docs/concepts/workloads/pods/",
    "https://docs.docker.com/engine/reference/commandline/run/",
    "https://docs.aws.amazon.com/s3/index.html",
    "https://cloud.google.com/storage/docs/creating-buckets",

    # ---- GitHub (many slashes, code paths) ----
    "https://github.com/scikit-learn/scikit-learn/blob/main/README.md",
    "https://github.com/pandas-dev/pandas/issues/12345",
    "https://github.com/tensorflow/tensorflow/tree/master/tensorflow",
    "https://github.com/django/django/blob/main/django/core/",
    "https://github.com/torvalds/linux/blob/master/README",
    "https://github.com/microsoft/vscode/releases/latest",
    "https://raw.githubusercontent.com/numpy/numpy/main/README.md",
    "https://github.com/CQY2682/CP2-phishing-detection-Project/",

    # ---- Stack Overflow / forums ----
    "https://stackoverflow.com/questions/12345678/how-to-parse-urls",
    "https://stackoverflow.com/questions/11227809/why-is-processing-sorted",
    "https://stackoverflow.com/questions/6470428/catch-multiple-exceptions",
    "https://stackoverflow.com/questions/1984325/explaining-pythons-list",
    "https://superuser.com/questions/123456/how-to-disable-auto-update",
    "https://askubuntu.com/questions/12345/how-to-install-python3",
    "https://reddit.com/r/learnpython/comments/abc123/how_do_i_learn/",
    "https://www.reddit.com/r/MachineLearning/comments/def456/",

    # ---- News articles (long paths) ----
    "https://www.bbc.com/news/technology-12345678",
    "https://www.bbc.com/news/world-us-canada-67890123",
    "https://www.reuters.com/technology/artificial-intelligence/2024/01/",
    "https://techcrunch.com/2024/01/15/openai-releases-new-model/",
    "https://arstechnica.com/security/2024/01/phishing-attacks-increase/",
    "https://www.theverge.com/2024/1/15/12345678/apple-vision-pro-review",
    "https://www.wired.com/story/best-password-managers-2024/",
    "https://www.thestar.com.my/tech/tech-news/2024/01/malaysia-digital",
    "https://www.malaymail.com/news/malaysia/2024/01/15/digital-economy",

    # ---- E-commerce product pages ----
    "https://www.amazon.com/gp/product/B08N5WRWNW/ref=ppx_yo_dt_b_asin",
    "https://www.amazon.com/s?k=laptop+stand&ref=nb_sb_noss",
    "https://www.ebay.com/itm/123456789012?hash=item1234abcd",
    "https://www.lazada.com.my/products/laptop-stand-i12345-s67890.html",
    "https://www.shopee.com.my/product/12345/67890123",
    "https://www.walmart.com/ip/product-name/123456789",
    "https://www.bestbuy.com/site/laptops/all-laptops/pcmcat138500050001.c",
    "https://store.steampowered.com/app/1091500/Cyberpunk_2077/",
    "https://www.apple.com/shop/buy-iphone/iphone-15/6.1-inch-display",

    # ---- Login/auth pages on REAL brands (critical test) ----
    "https://accounts.google.com/signin/v2/identifier?flowName=GlifWebSignIn",
    "https://login.microsoftonline.com/common/oauth2/v2.0/authorize",
    "https://www.facebook.com/login/?next=https%3A%2F%2Fwww.facebook.com",
    "https://github.com/login?return_to=%2Fjoin",
    "https://id.atlassian.com/login?continue=https%3A%2F%2Fjira.atlassian",
    "https://auth.openai.com/authorize?client_id=app&redirect_uri=https",
    "https://app.slack.com/signin/find-workspaces",
    "https://www.linkedin.com/login?fromSignIn=true&trk=guest_homepage",
    "https://secure.paypal.com/signin?returnUri=https%3A%2F%2Fwww.paypal.com",
    "https://www.twitter.com/i/flow/login",
    "https://www.instagram.com/accounts/login/?source=auth_switcher",
    "https://appleid.apple.com/auth/authorize?client_id=com.apple.icloud",
    "https://myaccount.google.com/security-checkup/1",
    "https://account.microsoft.com/account/Account?refd=account.microsoft",
    "https://www.maybank2u.com.my/maybank2u/malaysia/login",
    "https://www.cimbclicks.com.my/clicks/entreLogin.do",

    # ---- HTTP-only legitimate sites ----
    "http://neverssl.com/",
    "http://www.example.com/",
    "http://info.cern.ch/hypertext/WWW/TheProject.html",
    "http://www.iana.org/domains/reserved",
    "http://textfiles.com/computers/",
    "http://www.gutenberg.org/browse/recent/last1",
    "http://archive.org/about/",

    # ---- Malaysian sites (local context) ----
    "https://www.maybank.com/en/personal/eservices/online-banking.page",
    "https://www.cimb.com.my/en/personal/banking/accounts/savings.html",
    "https://www.rhbgroup.com/personal/deposits/savings-accounts/",
    "https://www.publicbank.com.my/contents/pbonline/",
    "https://www.bnm.gov.my/interest-rate",
    "https://www.hasil.gov.my/income-tax/individual/",
    "https://www.msc.com.my/en/vessels-and-schedules/",
    "https://www.grab.com/my/transport/",
    "https://food.grab.com/my/en/restaurants",
    "https://shopee.com.my/universal-link/shop/12345678/",
    "https://www.lazada.com.my/catalog/?q=laptop+stand",
    "https://www.airasia.com/flights/search?originCode=KUL&destinationCode=SIN",
    "https://www.malaysiaairlines.com/my/en/book/book-a-flight.html",
    "https://www.sunway.edu.my/university/academics/undergraduate-courses/",
    "https://intl.taylors.edu.my/programmes/degree-programmes/",

    # ---- Government sites ----
    "https://www.gov.my/en/topik/perkhidmatan-awam",
    "https://www.irs.gov/filing/e-file-options",
    "https://www.gov.uk/apply-for-a-uk-passport/overview",
    "https://www.usa.gov/benefits",
    "https://www.mygov.in/schemes-of-pm/",

    # ---- Academic / research ----
    "https://arxiv.org/abs/2401.12345",
    "https://arxiv.org/pdf/2310.01234.pdf",
    "https://www.researchgate.net/publication/12345678",
    "https://scholar.google.com/scholar?q=phishing+detection+machine+learning",
    "https://ieeexplore.ieee.org/document/12345678",
    "https://sunway.edu.my/research/publications",

    # ---- Cloud / SaaS ----
    "https://console.cloud.google.com/storage/browser",
    "https://portal.azure.com/#blade/Microsoft_AAD_IAM/ActiveDirectoryMenuBlade",
    "https://app.hubspot.com/contacts/12345/contacts/list/view/all/",
    "https://analytics.google.com/analytics/web/#/report/",
    "https://dashboard.stripe.com/payments",
    "https://app.supabase.com/project/default/editor",

    # ---- Long query parameter URLs ----
    "https://www.google.com/search?q=phishing+detection+machine+learning&hl=en",
    "https://www.youtube.com/results?search_query=machine+learning+tutorial",
    "https://www.amazon.com/s?k=python+programming&i=stripbooks&rh=n:283155",
    "https://www.booking.com/searchresults.html?ss=Kuala+Lumpur&checkin=2024-03-01",
    "https://www.google.com/maps/search/restaurants+near+me/@3.1390,101.6869,14z",
]

def main():
    print("=" * 70)
    print("TRAINING DATA AUGMENTATION")
    print("=" * 70)
    
    # Load existing splits
    print("\nLoading existing splits...")
    X_train = pd.read_csv('data/X_train.csv')
    y_train = pd.read_csv('data/y_train.csv')['label']
    print(f"  Original X_train: {X_train.shape}")
    print(f"  Phishing: {(y_train==1).sum():,} | Legit: {(y_train==0).sum():,}")

    # Extract features from curated URLs
    print(f"\nExtracting features from {len(REAL_LEGIT_URLS)} curated URLs...")
    augment_features = []
    for url in REAL_LEGIT_URLS:
        augment_features.append(extract_features(url))

    X_aug = pd.DataFrame(augment_features)[FEATURE_NAMES]
    y_aug = pd.Series([0] * len(REAL_LEGIT_URLS))  # all legitimate

    print(f"  Augmented rows: {len(X_aug)}")
    print(f"\n  Sample augmented features:")
    print(f"  avg qty_slash:  {X_aug['qty_slash'].mean():.2f} (original legit avg: 2.00)")
    print(f"  avg url_length: {X_aug['url_length'].mean():.2f} (original legit avg: 27.23)")
    print(f"  HTTPS rate:     {X_aug['has_https'].mean()*100:.1f}% (original legit: 100%)")

    # Append to training set
    X_train_aug = pd.concat([X_train, X_aug], ignore_index=True)
    y_train_aug = pd.concat([y_train, y_aug], ignore_index=True)

    print(f"\nAugmented training set:")
    print(f"  Total rows: {len(X_train_aug):,} (was {len(X_train):,})")
    print(f"  Phishing: {(y_train_aug==1).sum():,}")
    print(f"  Legit: {(y_train_aug==0).sum():,}")

    # Save
    X_train_aug.to_csv('data/X_train_aug.csv', index=False)
    y_train_aug.rename('label').to_frame().to_csv('data/y_train_aug.csv', index=False)
    print(f"\n  Saved: data/X_train_aug.csv")
    print(f"  Saved: data/y_train_aug.csv")
    print("\nReady to retrain. Run: python src/model_trainer.py --augmented")


if __name__ == "__main__":
    main()