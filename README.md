### **Περιγραφή των αρχείων**

1. **`text_processing.py`**
   - Περιέχει συναρτήσεις για τη φόρτωση και επεξεργασία του dataset.
   - Υλοποιεί την αφαίρεση ειδικών χαρακτήρων, την αφαίρεση stopwords, και τη λεμματοποίηση.
   - Επεξεργάζεται ολόκληρο το dataset και το αποθηκεύει σε JSON μορφή.

2. **`inverted_index.py`**
   - Δημιουργεί ένα ανεστραμμένο ευρετήριο από το επεξεργασμένο dataset.
   - Αποθηκεύει το ευρετήριο σε αρχείο JSON για χρήση από άλλες λειτουργίες.

3. **`search_engine.py`**
   - Παρέχει βασικές λειτουργίες αναζήτησης μέσω του ανεστραμμένου ευρετηρίου.
   - Υλοποιεί διεπαφή χρήστη για αναζήτηση ερωτημάτων και επιστρέφει τα αποτελέσματα.

4. **`query_processing.py`**
   - Υποστηρίζει σύνθετα Boolean queries (AND, OR, NOT).
   - Παρέχει συναρτήσεις για την κανονικοποίηση και την επεξεργασία ερωτημάτων.
   - Περιλαμβάνει διεπαφή χρήστη για την εισαγωγή και εκτέλεση Boolean queries.

5. **`ranking.py`**
   - Υλοποιεί τρεις αλγορίθμους κατάταξης: TF-IDF, Vector Space Model (VSM), και BM25.
   - Κατατάσσει τα αποτελέσματα αναζήτησης με βάση τη σημασία τους.
   - Παρέχει διεπαφή χρήστη για την επιλογή αλγορίθμου και την εμφάνιση αποτελεσμάτων.

6. **`main.py`**
   - Οργανώνει τη ροή της εφαρμογής:
     1. Φορτώνει και επεξεργάζεται τα δεδομένα.
     2. Δημιουργεί το ανεστραμμένο ευρετήριο.
     3. Εκτελεί αναζητήσεις.
     4. Κατατάσσει τα αποτελέσματα και υπολογίζει μετρικές αξιολόγησης (Precision, Recall, F1-Score).
   - Συγκεντρώνει όλες τις λειτουργίες σε μία ολοκληρωμένη ροή.

7. **`goalscorers.json`**
   - Το αρχικό dataset με δεδομένα για σκορ και σκόρερ από αγώνες ποδοσφαίρου.

8. **`goalscorers_processed.json`**
   - Το επεξεργασμένο dataset μετά την εφαρμογή των τεχνικών προεπεξεργασίας.

9. **`inverted_index.json`**
   - Το ανεστραμμένο ευρετήριο που δημιουργήθηκε από τα δεδομένα.
