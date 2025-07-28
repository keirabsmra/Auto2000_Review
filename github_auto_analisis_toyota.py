import os
import json
import pandas as pd
import numpy as np
import re
import joblib
import emoji
import unicodedata
import gdown
from collections import Counter
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import gspread
from google.oauth2.service_account import Credentials

"""### **SETUP GOOGLE SHEETS**"""

SERVICE_ACCOUNT = os.environ["GOOGLE_CREDENTIALS_JSON"]
SERVICE_DICT = json.loads(SERVICE_ACCOUNT)

scopes = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]

creds = Credentials.from_service_account_info(SERVICE_DICT, scopes=scopes)
gc = gspread.authorize(creds)

sheet = gc.open("Review Auto2000").worksheet("Sheet1")
data = sheet.get_all_records()
df = pd.DataFrame(data)

model_dir = "models"
os.makedirs(model_dir, exist_ok=True)

xgb_url = "https://drive.google.com/uc?id=1xBJQcqn-T9-XAC_9EDFe4_jp1_im5fZF"
tfidf_url = "https://drive.google.com/uc?id=1amEY_RJVR70z0pEa40qDTTLvbXuXo0g0"

xgb_path = f"{model_dir}/new_xgboost_sentiment_model.pkl"
tfidf_path = f"{model_dir}/new_tfidf_vectorizer.pkl"

gdown.download(xgb_url, xgb_path, quiet=False)
gdown.download(tfidf_url, tfidf_path, quiet=False)

vectorizer = joblib.load(tfidf_path)
xgb_model = joblib.load(xgb_path)

"""### **Data Preparation**"""

def detect_cabang(address):
    if pd.isna(address):
        return 'Unknown'
    address = address.lower()
    if 'bsd raya utama' in address:
        return 'Auto2000 BSD City'
    elif 'komp. bsd' in address:
        return 'Auto2000 BSD'
    elif 'pondok jagung' in address or 'km.7' in address:
        return 'Auto2000 Pondok Jagung'
    elif 'alam sutera' in address or 'jalur sutera' in address:
        return 'Auto2000 Alam Sutera'
    elif 'moh. husni thamrin' in address or 'sektor vii' in address:
        return 'Auto2000 Bintaro'
    elif 'rc. veteran raya' in address:
        return 'Tunas Toyota Bintaro'
    elif 'raya boulevard' in address:
        return 'Plaza Toyota Gading Serpong'
    else:
        return 'Unknown'

def prepare_data(df_input):
    df = df_input.copy()
    df.columns = df.columns.str.strip()
    available_cols = df.columns.tolist()

    base_cols = {
        'publishedAtDate': 'review_date',
        'text': 'review_text',
        'stars': 'review_star',
        'likesCount': 'review_likes',
        'address': 'review_address'
    }

    selected_cols = [col for col in base_cols if col in available_cols]
    photo_cols = [col for col in available_cols if col.startswith('reviewImageUrls/')]

    final_cols = selected_cols + photo_cols
    df_clean = df[final_cols].copy()

    # Rename kolom
    df_clean.rename(columns={k: v for k, v in base_cols.items() if k in df_clean.columns}, inplace=True)

    # Deteksi foto
    if photo_cols:
        df_clean['has_photo'] = df_clean[photo_cols].applymap(lambda x: x not in [None, "", "NaN"]).any(axis=1).astype(int)
        df_clean.drop(columns=photo_cols, inplace=True)
    else:
        df_clean['has_photo'] = 0

    # Deteksi cabang
    if 'review_address' in df_clean.columns:
        df_clean['cabang'] = df_clean['review_address'].apply(detect_cabang)
    else:
        df_clean['cabang'] = 'Unknown'

    return df_clean

# review kosong
df_clean = prepare_data(df)
mask_kosong = df_clean['review_text'].apply(lambda x: pd.isna(x) or str(x).strip() == '')
data_kosong = df_clean[mask_kosong]

df_clean = df_clean[df_clean['review_text'].notna() & (df_clean['review_text'].str.strip() != "")]

"""### **Text Prepocessing**"""

import pandas as pd
import re
import string
import emoji
import unicodedata
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# Load slang dictionary
slangs = pd.read_csv("https://raw.githubusercontent.com/nasalsabila/kamus-alay/master/colloquial-indonesian-lexicon.csv")
slang_dict = dict(zip(slangs['slang'], slangs['formal']))

# Stopwords
stop_factory = StopWordRemoverFactory()
sastrawi_stopwords = set(stop_factory.get_stop_words())

custom_stopwords = {
    'nya', 'mas', 'mba', 'mbak', 'pak', 'kak', 'apa', 'sama', 'the', 'yang', 'dan', 'atau', 'jadi', 'dll', 'banget',
    'sebagai', 'untuk', 'juga', 'dengan', 'oleh', 'sih', 'kok', 'deh', 'lagi', 'sangat', 'â',
    'di', 'ke', 'of', 'by', 'in', 'it', 'is', 'am', 'are', 'you', 'we', 'us', 'our', 'and'
}
stop_words = sastrawi_stopwords.union(custom_stopwords)
stop_words.discard('ada')
stop_words.discard('tidak')

# Koreksi typo
typo_corrections = {
    'rsmah': 'ramah', 'servcie': 'service', 'peyanan': 'pelayanan', 'scuritynya': 'security',
    'tempat\'a': 'tempatnya', 'nungu': 'nunggu', 'oil': 'oli', 'playanan': 'pelayanan', 'pasilitas': 'fasilitas',
    'servuce': 'service', 'mzshollanya': 'musholanya', 'pelaianan': 'pelayanan', 'mbl': 'mobil', 'escram': 'eskrim',
    'pelanyananya': 'pelayanannya', 'pekerjaanya': 'pekerjaannya', 'pelayananbyang': 'pelayanan yang',
    'istimasikan': 'estimasikan', 'kelluhn': 'keluhan', 'biyaya': 'biaya', 'sete': 'setelah', 'palayanan': 'pelayanan',
    'trnyata': 'ternyata', 'sadah': 'sudah', 'ahir': 'akhir', 'husus': 'khusus', 'eskram': 'eskrim',
    'pengin': 'pengen', 'alhamdulil lah': 'alhamdulillah', 'dibannyakin': 'dibanyakin', 'pekayanan': 'pelayanan',
    'menyennagkan': 'menyenangkan', 'castamer': 'customer', 'cerbis': 'servis', 'terpertjaja': 'terpercaya',
    'survay': 'survey', 'bahus': 'bagus', 'revristrasi': 'registrasi', 'pmbelian': 'pembelian', 'parahbsich': 'parah'
}
def correct_typos(text):
    for typo, correct in typo_corrections.items():
        text = text.replace(typo, correct)
    return text

def remove_emoji(text):
    if pd.isnull(text):
        return ''
    text = emoji.replace_emoji(text, replace='')
    emoji_pattern = re.compile(
        "["u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        u"\U00002700-\U000027BF"
        u"\U0001F900-\U0001F9FF"
        u"\U00002600-\U000026FF"
        "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def normalize_unicode(text):
    return ''.join(
        c for c in unicodedata.normalize('NFKD', str(text))
        if not unicodedata.combining(c) and ord(c) < 128
    )

def reduce_repeated_chars(text):
    return re.sub(r'([a-zA-Z])\1{2,}', r'\1', text)

def prepo_slang(text):
    return ' '.join([slang_dict.get(word, word) for word in text.split()])

# Fungsi utama preprocessing
def preprocess_review(text):
    text = str(text).lower()
    text = normalize_unicode(text)
    text = correct_typos(text)
    text = reduce_repeated_chars(text)
    text = remove_emoji(text)

    # Proteksi angka jam
    text = re.sub(r'(\d+)\.(\d+)', r'\1DOT\2', text)
    text = re.sub(r'\n|\t', ' ', text)

    # Ganti tanda hubung dengan spasi
    text = re.sub(r'[-]', ' ', text)

    # Pisahkan akhiran
    text = re.sub(r'([a-z]+)(nya)\b', r'\1 \2', text)
    text = re.sub(r'\.{2,}', ' ', text)
    text = re.sub(r'[.,;:/\\]', ' ', text)

    # Hapus simbol
    text = re.sub(r'[^\w\s]', '', text)
    text = ' '.join(text.split())

    # Slang
    text = prepo_slang(text)
    text = text.replace('DOT', '.')

    # Stopword
    words = [word for word in text.split() if word not in stop_words]
    return ' '.join(words)

df_clean['cleaned_review'] = df_clean['review_text'].apply(preprocess_review)
df_clean = df_clean[df_clean['cleaned_review'].str.strip() != '']

df_clean = df_clean[
    ~df_clean['review_text'].str.contains('[äöüßÄÖÜ]', regex=True) &
    ~df_clean['review_text'].str.contains('[ㄱ-ㅎㅏ-ㅣ가-힣]', regex=True) &
    ~df_clean['review_text'].str.contains('nakeuh|gampông|peudada', case=False) &
    ~df_clean['review_text'].str.contains('[\u4e00-\u9fff]', regex=True)]
df_clean.reset_index(drop=True, inplace=True)

def contains_non_ascii(text):
    return any(ord(char) > 127 for char in str(text))

df_clean = df_clean[~df_clean['review_text'].apply(contains_non_ascii)]
df_clean.reset_index(drop=True, inplace=True)
# print(f"\nJumlah total review setelah preprocessing: {df_clean.shape[0]} baris")

# Transform cleaned review menjadi fitur numerik
X_new = vectorizer.transform(df_clean['cleaned_review'])

# Prediksi sentimen
df_clean['predicted_sentimen'] = xgb_model.predict(X_new)

# Mapping label
sentimen_mapping = {0: 'netral', 1: 'positif', 2: 'negatif'}
df_clean['sentimen_label'] = df_clean['predicted_sentimen'].map(sentimen_mapping)

# Rule based adjustment
positive_keywords = ['ramah', 'baik', 'puas', 'cepat', 'mantap', 'bagus', 'rekomendasi', 'langganan', 'bersih', 'memuaskan', 'profesional', 'berkualitas', 'terpercaya', 'good',
                     'teroganizir', 'sukses', 'aman', 'great service', 'terimakasih', 'bantuan pelayanan']
negative_keywords = ['lama', 'jelek', 'parah', 'buruk', 'mengecewakan', 'telat', 'masalah', 'bising', 'salah', 'curang', 'kurang', 'jengkel', 'enggak jelas', 'kecewa', 'cacat',
                     'asal asalan', 'unprofessional', 'susah hubungi', 'kurang puas', 'lambat', 'lemot', 'tidak teratasi', 'sulit sekali', 'guobkok', 'kurang ramah', 'tidak simetris',
                     'tidak responsif', 'tidak pernah diangkatt', 'poor', 'tidak memuaskan', 'terburuk', 'kecewa', 'kurang puas', 'songong', 'payah', 'kurang menyenangkan', 'ribet',
                     'tidak recommend', 'tidak lazim', 'cukup kecewa', 'terpaksa', 'tanggapan kurang', 'susah', 'buang waktu', 'tidak jelas', 'sulit sekali', 'pelayanan jelek',
                     'enggak ada melayani', 'kualitas menurun', 'busuk pelayanan', 'bad', 'very bad', 'tidak memuaskan', 'pengalaman kurang']

# Adjustment function
def adjust_sentimen(row):
    text = row['cleaned_review'].lower()
    if row['sentimen_label'] == 'netral':
        if any(kw in text for kw in positive_keywords):
            return 'positif'
        elif any(kw in text for kw in negative_keywords):
            return 'negatif'
    return row['sentimen_label']

df_clean['adjusted_sentimen'] = df_clean.apply(adjust_sentimen, axis=1)
df_clean['sentimen_label'] = df_clean['adjusted_sentimen']

"""### **Rule based text classification**

#### Kategori
"""

# Keyword Sales
sales_keywords = [
    'sales', 'salesnya', 'salesperson', 'marketing', 'buy', 'buy car', 'bought', 'unit toyota', 'terimakasih sales',
    'pembelian', 'pembelian mobil', 'pembelian toyota', 'servicesales', 'pelayanan sales', 'pelayanan salesnya', 'pemesanan mobil',
    'beli', 'beli mobil', 'beli toyota', 'beli unit', 'membeli mobil', 'membeli toyota', 'order', 'repeat order',
    'unit', 'unit diantar', 'pengiriman', 'pengiriman cepat', 'pengiriman unit', 'dp', 'proses beli', 'serah terima',
    'test drive', 'consultant mobil', 'fasilitas mobil', 'sosialisasi', 'stnk', 'plat', 'mobil datang', 'for buy', 'sales ramah',
    'cari mobil', 'ambil mobil', 'ambil toyota', 'great sales', 'penyediaan unit', 'ambil unit', 'pelayanan dealer', 'pembelian pertama',
    'delivery unit', 'pengalaman membeli', 'angsuran', 'pilihan used car', 'varian mobil', 'asuransi', 'ganti mobil', 'beli yaris cross',
    'mengantarkan mobil', 'proper recommendation', 'trade in', 'trade mobil', 'memiliki mobil', 'mobil impian', 'cooperative support',
    'penawaran', 'pengantaran', 'sale service center', 'proses pembelian', 'melayani penjualan', 'kendaraan baru', 'delivery mobil',
    'process buying', 'pesan mobil', 'mobil diterima', 'beli mobil', 'beli mobil baru', 'mobil dipesan', 'proses acc'
]

# Keyword Aftersales
aftersales_keywords = [
    'booking', 'booking service', 'jadwal service', 'tunggu', 'nunggu', 'waiting', 'tunggunya', 'parking area', 'ruang', 'puas', 'eskrim',
    'ruang tunggu', 'pelayana', 'pelayan', 'bengkel', 'advisor', 'oli', 'ganti oli', 'menunggu', 'berkala', 'antrian', 'pelaya', 'cofee',
    'serviced', 'penanganan', 'hasil', 'layanan', 'booked', 'jam', 'makan', 'service berkala', 'servis berkala', 'montir', 'on time',
    'mekanik', 'perawatan', 'sa', 'teknisi', 'estimasi', 'free coffe', 'pelayanan cepat', 'sparepart', 'sparepartnya', 'repaint', 'telpon',
    'spare part', 'pelayanan', 'service', 'servis', 'security', 'satpam', 'services', 'keluhan', 'fasilitas','voucher', 'ganti panel',
    'first crack', 'ruang tunggu', 'kopi', 'coffee', 'cafe', 'tempat tunggu', 'servicenya', 'tunggu servis', 'disambut', 'luas', 'permasalahan mobil',
    'body repair', 'rapi', 'charger mobil', 'well managed automotive', 'penanganan cepat', 'perbaikan', 'boking', 'nonboking', 'servic mobil',
    'helpful', 'sigap', 'body paint', 'parking', 'repair toyota', 'check up', 'lumayan cepat', 'helpfull', 'lincah', 'mobil rusak',
    'mobil bagus', 'auto vehicle', 'serve', 'cukup puas', 'telepon', 'call center', 'competent', 'masalah rem', 'telp kantor', 'suku cadang',
    'pengecheck kendaraan', 'fast response', 'telp', 'penanganan yang cepat', 'pengerjaan cepat', 'pekerjaan bagus', 'cekatan', 'jaminan mutu',
    'teliti', 'petugas keamanan', 'working very proffesional', 'staff ramah', 'good apps', 'technician', 'pengerjaan lama', 'maintenance',
    'nyaman', 'adem', 'bagus', 'book', 'cozy', 'dingin', 'sovenir', 'bersih', 'well manage', 'proses claim', 'cepat kerjanya', 'makanan siang',
    'ramah staff', 'steer', 'selesai sesuai estimasi', 'frontliner', 'servise', 'makananan', 'snack', 'enggak kerjain', 'dicuci', 'sporing ban',
    'quickly', 'kantin', 'smoking area', 'pelayanan ramah', 'bemper depan mobil', 'setir', 'pegawai ramah', 'repair', 'repair mobil', 'spion',
    'amplas kovax', 'body penyok', 'merefair mobil', 'car repair', 'nice staff', 'cepat', 'gesit', 'melayani customer', 'ban kempes', 'cs melayani',
    'proses recall', 'call', 'equipments', 'technicians', 'parts', 'melayani', 'sambutan', 'helping', 'nomor mesin', 'analisa ulang', 'eskrim gratis',
    'tutup ban', 'body clip', 'lem silikon', 'ngasal pekerjaan', 'kecepatan bekerja', 'tune up', 'regular tune up', 'memenuhi kebutuhan', 'poles',
    'menyejukan', 'petugas profesional', 'slow respon', 'fast respon', 'rusak', 'susah hubungi', 'biaya jasa', 'customer friendly', 'ramah customer'
]

# Ambiguous Words
ambiguous_keywords = ['nyaman', 'ramah', 'cepat', 'bagus', 'baik']

# Nama sales untuk deteksi tambahan
sales_names = {
    'alfaro', 'hamar', 'beryl', 'aldo', 'tobar', 'dila', 'gifari', 'tian', 'alfero', 'aditya', 'warman', 'hamzah', 'adit warman',
    'syahlian', 'maya', 'fabian', 'alvaro', 'daffa', 'wawan', 'agus', 'alfaroh', 'rusmawan', 'berryl', 'anggi', 'adit'
}

# Fungsi klasifikasi kategori review
def assign_kategori(row):
    text = str(row['cleaned_review']).lower()
    tokens = set(text.split())
    cabang = str(row['cabang']).strip().lower()

    # Pisahkan keyword satu dan dua kata
    sales_1word = [kw for kw in sales_keywords if ' ' not in kw]
    sales_2word = [kw for kw in sales_keywords if ' ' in kw]
    aftersales_1word = [kw for kw in aftersales_keywords if ' ' not in kw]
    aftersales_2word = [kw for kw in aftersales_keywords if ' ' in kw]

    # Hitung keyword match
    sales_hits = sum(kw in tokens for kw in sales_1word) + sum(kw in text for kw in sales_2word)
    after_hits = sum(kw in tokens for kw in aftersales_1word) + sum(kw in text for kw in aftersales_2word)

    # Nama sales hanya berlaku untuk Auto2000 BSD City
    if 'auto2000 bsd city' in cabang: # Use the 'cabang' from the row
        if any(name in tokens for name in sales_names):
            sales_hits += 1

    # Konteks "pelayanan"
    if 'pelayanan' in tokens:
        if 'auto2000 bsd city' in cabang and any(name in tokens for name in sales_names): # Use the 'cabang' from the row
            sales_hits += 1
        elif any(kw in text for kw in ['ruang tunggu', 'service', 'advisor', 'bengkel', 'voucher']):
            after_hits += 1

    # Logika akhir
    if sales_hits == 0 and after_hits == 0:
        return 'Other'
    elif sales_hits > after_hits:
        return 'Sales'
    elif after_hits > sales_hits:
        return 'Aftersales'
    else:
        return 'Sales'

# Apply the function row-wise
df_clean['kategori_review'] = df_clean.apply(assign_kategori, axis=1)

# persentase tiap kategori
kategori_counts = df_clean['kategori_review'].value_counts()
kategori_percentages = kategori_counts / len(df_clean) * 100

"""#### Multiaspek"""

# Aspect keywords dictionary
aspect_keywords = {
    'Sales': [
        'sales', 'salesnya', 'pembelian', 'unit', 'beli', 'delivery', 'serah terima', 'pengiriman', 'beli mobil',
        'test drive', 'consultant', 'buy', 'buy car', 'bought', 'salesperson', 'pembelian mobil', 'pengiriman unit',
        'pelayanan sales', 'order', 'repeat order', 'ambil unit', 'mobil baru', 'unit baru', 'salesman'
    ],
    'Booking': [
        'booking', 'booked', 'jadwal', 'whatsapp', 'telpon', 'telepon', 'reschedule', 'appointment', 'tutup',
        'ubah jadwal', 'batal booking', 'gagal booking', 'antrian', 'janji', 'via', 'online', 'info', 'jadwal service',
        'nonboking', 'nonbooking', 'boking', 'slow respon', 'fast respon', 'susah hubungi', 'telp', 'call', 'telefon'
    ],
    'Fasilitas': [
        'ruang tunggu', 'lounge', 'kopi', 'coffee', 'first crack', 'makan', 'sarapan', 'voucher', 'free coffe', 'makan siang',
        'tempat', 'tempatnya', 'cozy', 'wifi', 'ac', 'area', 'kursi', 'toilet', 'mushola', 'musholla', 'kupon', 'parkiran',
        'waiting room', 'snack', 'minuman', 'air minum', 'parkir', 'satpam', 'fasilitas', 'dealer', 'disambut', 'mewah', 'security',
        'bengkel bagus', 'facilities', 'free', 'environment', 'gratis makan', 'free lunch', 'nyaman', 'place', 'comfortable',
        'pantries', 'neighborhood', 'luas', 'parking area', 'ruang',  'nyaman', 'adem', 'book', 'cozy', 'showroom',
        'dingin', 'sovenir', 'bersih', 'makanan siang', 'kantin', 'smoking area', 'bingkisan', 'strategis', 'comfort', 'serve breakfast',
        'lunch', 'lokasi', 'cofee', 'cleanliness'
    ],
    'Teknisi': [
        'teknisi', 'mekanik', 'montir', 'perbaikan', 'pemeriksaan', 'hasil pekerjaan', 'tidak ditemukan masalah', 'speed service', 'service good',
        'kerjanya rapih', 'kerjanya bersih', 'kualitas', 'handal', 'perbaiki', 'diagnosa', 'kurang pengalaman', 'sudah dibenerin', 'great service',
        'good service', 'servis bagus', 'hasil', 'diatasi',  'oli', 'ganti oli', 'montir', 'body paint', 'repair toyota', 'check up', 'good servis',
        'mobil rusak', 'mobil bagus', 'masalah rem', 'pengecheck kendaraan',  'penanganan yang cepat', 'pengerjaan cepat', 'pekerjaan bagus',
        'cekatan', 'technician', 'pengerjaan lama', 'cepat kerjanya', 'steer', 'enggak kerjain', 'dicuci', 'sporing ban', 'bemper depan mobil',
        'setir', 'repair', 'repair mobil', 'spion',  'body penyok', 'ban kempes', 'hasil servis', 'tidak maksimal', 'kurang rapi',
        'tune up', 'regular tune up', 'nice service', 'best service', 'well trained', 'well manage', 'technicians', 'gps problem', 'not fixed',
        'servis mobil', 'service mobil', 'service excellent', 'fast service', 'tindakan teknis', 'perawatan mobil', 'tangani baik'
    ],
    'Harga': [
        'harga', 'biaya', 'tarif', 'biaya jasa', 'garansi', 'promo', 'diskon', 'potongan', 'mahal', 'murah', 'kredit', 'dp',
        'harga sesuai', 'biaya tidak dijelaskan', 'juta', 'ratus', 'biaya jasa', 'biaya pengecekan', 'waranty', 'price'
    ],
    'Waktu': [
        'jam', 'waktu', 'lama', 'estimasi', 'delay', 'molor', 'cepat', 'menunggu', 'prosesnya',
        'datang', 'selesai', 'senin', 'selasa', 'rabu', 'kamis', 'jumat', 'sabtu', 'minggu', 'waiting',
        'weekday', 'weekend', 'since', 'tidak cepat', 'antre', 'lama sekali', 'waktu pengerjaan', 'proses cepat',
        'waktu tunggu', 'terlalu lama', 'jam buka', 'jam kerja', 'lambat', 'lemot'
    ],
    'Sparepart': [
        'sparepart', 'spare part', 'spart', 'tmo', 'onderdil', 'suku cadang', 'stok', 'pesan spart', 'barang pending', 'gak ada part',
        'barang tidak ada', 'sparepart kosong', 'pending', 'stok sparepart', 'oli', 'wiper', 'barang', 'mesin', 'indent', 'parts',
        'amplas kovax', 'equipments', 'stok barang', 'barang ready', 'ketersediaan barang', 'pre order', 'bengkel lengkap'
    ],
    'SA': [
        'sa', 'advisor', 'service advisor', 'staff', 'petugas', 'alip', 'hery', 'herry',
        'responsif', 'komunikatif', 'friendly', 'informatif', 'helpful', 'informasi', 'bagus pelayana'
        'penjelasan', 'sopan', 'ramah', 'penanganan', 'jelas', 'cs', 'service cs', 'pelayan',
        'customer services', 'best customer service', 'pelayanan bagus', 'pelayanan mantap',
        'pelayanan', 'tidak diinformasikan', 'pelayana', 'fast personalized'
    ]
}
aspect_priority = [
    'Sparepart',
    'Harga',
    'Booking',
    'Teknisi',
    'Waktu',
    'SA',
    'Fasilitas',
    'Sales']

def count_keyword_hits(text, keywords):
    count = 0
    for kw in keywords:
        if ' ' in kw:
            if kw in text:
                count += 1
        else:
            if kw in text.split():
                count += 1
    return count

def assign_multi_aspects_v3(row):
    text = str(row['cleaned_review']).lower()
    category = row['kategori_review'].strip().lower()

    if category == 'sales':
        return ['Sales', 'Sales', 'Sales']

    # Skoring aspek
    aspect_scores = {}
    for aspect in aspect_priority:
        if aspect == 'Sales':
            continue
        keywords = aspect_keywords.get(aspect, [])
        score = count_keyword_hits(text, keywords)
        if score > 0:
            aspect_scores[aspect] = score

    # Urutkan skor + prioritas
    sorted_aspects = sorted(
        aspect_scores.items(),
        key=lambda x: (-x[1], aspect_priority.index(x[0]))
    )
    aspects = [aspect for aspect, _ in sorted_aspects]

    if len(aspects) == 0:
        return ['Other', 'Other', 'Other']
    elif len(aspects) == 1:
        return aspects * 3
    elif len(aspects) == 2:
        return aspects + [aspects[0]]
    else:
        return aspects[:3]

df_clean[['aspect_1', 'aspect_2', 'aspect_3']] = df_clean.apply(
    assign_multi_aspects_v3, axis=1, result_type='expand')

# debugging aspek
def debug_aspect_hits(text, aspect_keywords):
    text = str(text).lower()
    result = {}
    for aspect, keywords in aspect_keywords.items():
        matched = []
        for kw in keywords:
            if ' ' in kw:
                if kw in text:
                    matched.append(kw)
            else:
                if kw in text.split():
                    matched.append(kw)
        if matched:
            result[aspect] = matched
    return result

# # Distribusi Aspect
from collections import Counter
all_aspek = (
    df_clean['aspect_1'].tolist()
    + df_clean['aspect_2'].tolist()
    + df_clean['aspect_3'].tolist()
)
aspek_counts = Counter(all_aspek)  # ini penting, harus ada sebelum total_aspek_mentions
total_aspek_mentions = sum(aspek_counts.values())
aspek_percentages = {aspek: (count / total_aspek_mentions) * 100 for aspek, count in aspek_counts.items()}
sorted_aspek_percentages = dict(sorted(aspek_percentages.items(), key=lambda item: item[1], reverse=True))

"""#### Sentimen Bintang"""

def cek_konsistensi(star, sentiment):
    if star >= 4 and sentiment == 1:    #bintang 4/5 = positif
        return 'Konsisten'              #Rating tinggi → cocok dengan sentimen positif
    elif star <= 2 and sentiment == 2:  #bintang 2/1 = negatif
        return 'Konsisten'              #Rating rendah → cocok dengan sentimen negatif
    elif star == 3 and sentiment == 0:  #bintang 3 = netral
        return 'Konsisten'              #Rating tengah → cocok dengan sentimen netral
    else:
        return 'Tidak Konsisten'        #Jika sentimen tidak sesuai dengan ekspektasi rating

# Convert 'review_star' to numeric, coercing errors to NaN, then fill NaN with a placeholder if needed
df_clean['review_star'] = pd.to_numeric(df_clean['review_star'], errors='coerce')

df_clean['konsistensi_sentimen_rating'] = df_clean.apply(
    lambda row: cek_konsistensi(row['review_star'], row['predicted_sentimen']), axis=1)

# Jumlah dan persentase
counts = df_clean['konsistensi_sentimen_rating'].value_counts()
percentages = df_clean['konsistensi_sentimen_rating'].value_counts(normalize=True) * 100

konsistensi_summary = pd.concat([counts, percentages], axis=1)
konsistensi_summary.columns = ['Jumlah', 'Persentase (%)']

# Tambahkan kolom tahun dan sentimen numerik ke df_clean
df_clean['review_year'] = pd.to_datetime(df_clean['review_date'], errors='coerce').dt.year.astype('Int64')

sentiment_numeric_mapping = {'positif': 1, 'negatif': 2, 'netral': 0}
df_clean['sentimen'] = df_clean['sentimen_label'].map(sentiment_numeric_mapping).astype('Int64')

# Pilih kolom yang akan diekspor
df_export = df_clean[[
    'review_year', 'review_date', 'cabang', 'review_star', 'has_photo', 'cleaned_review',
    'sentimen', 'sentimen_label', 'kategori_review', 'aspect_1', 'aspect_2', 'aspect_3'
]].copy()

# Konversi numerik agar tidak ada desimal
df_export['review_year'] = df_export['review_year'].astype('Int64')
df_export['sentimen'] = df_export['sentimen'].astype('Int64')
df_export['review_star'] = df_export['review_star'].astype('Int64')

# Pastikan review_date dalam format datetime
df_export['review_date'] = pd.to_datetime(df_export['review_date'], errors='coerce')
df_export['review_date'] = df_export['review_date'].dt.tz_localize(None)

# Ubah datetime menjadi string agar aman untuk Google Sheets/Excel
for col in df_export.select_dtypes(include=['datetime64[ns]']).columns:
    df_export[col] = df_export[col].astype(str)

# === Simpan ke Google Sheets ===
spreadsheet_key = "1AKrwoAq7K-zXe-9L2VHB3j_d529Ot62VeNrDoykbcnQ"
spreadsheet = gc.open_by_key(spreadsheet_key)

try:
    sheet_out = spreadsheet.worksheet("Hasil Analisis")
except:
    sheet_out = spreadsheet.add_worksheet(title="Hasil Analisis", rows=1000, cols=20)

sheet_out.clear()
sheet_out.update([df_export_upload.columns.tolist()] + df_export_upload.values.tolist())
