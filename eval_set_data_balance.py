"""eval_set_data_balance.py - penambahan pertanyaan untuk menyeimbangkan dataset.

Dua masalah yang diperbaiki modul ini:

1. Sebaran question_type sangat timpang. Pada versi 350 pertanyaan, empat
   kategori punya n < 15 (komparatif 2, unanswerable 8, multi-hop 10,
   prosedur 13) sehingga tidak cukup untuk klaim statistik per kategori.
   Modul ini menaikkan keempatnya ke n >= 35.

2. Tiga belas dokumen baru masuk ke corpus tanpa satu pun pertanyaan gold,
   sehingga terbaca sebagai distractor padahal isinya justru paling dekat
   dengan pertanyaan mahasiswa sehari-hari (pendaftaran, daftar ulang,
   PKKMB, kukerta, panduan penelitian, panduan akademik FK).

ATURAN "YANG BERLAKU"
---------------------
Jawaban acuan selalu diambil dari dokumen yang MASIH BERLAKU. Dokumen yang
sudah dicabut atau digantikan sengaja TIDAK dipakai sebagai gold dan tetap
berada di corpus sebagai distractor sulit (isinya mirip tapi salah zaman):

  - Panduan Penelitian & Pengabdian 2025 (Kepmen 1694)  -> diganti versi 2026
  - Panduan Kukerta Reguler & MBKM 2023                 -> diganti Panduan 2026
  - Permenristekdikti 5/2016 (SSBO PTN BH)              -> dicabut Permen 30/2019
  - Permenristekdikti 39/2017 (BKT dan UKT)             -> sudah digantikan
  - Pertor 3/2015 dan Pertor 4/2021 (peraturan akademik lama)

Format baris sama dengan modul lain:
    (pertanyaan, jawaban_acuan, kode_gold, tipe, kesulitan, answerable)

Kode gold memakai indeks halaman 0-based agar cocok dengan metadata 'page'
dari PyPDFLoader.
"""

from __future__ import annotations

# Dokumen baru yang dipakai sebagai sumber jawaban.
DOCS_BAL = {
    "KKN":   "KEPUTUSAN REKTOR UNIVERSITAS RIAU TENTANG PANDUAN KULIAH KERJA NYATA BERDAMPAK UNIVERSITAS RIAU TAHUN 2026.pdf",
    "PKM":   "KEPUTUSAN REKTOR UNIVERSITAS RIAU TENTANG PANDUAN PENELITIAN DAN PENGABDIAN KEPADA MASYARAKAT.pdf",
    "IJZ22": "2022permendikbudristek006.pdf",
    "GRAT":  "2019permendikbud029.pdf",
    "SSBO":  "2019permendikbud030.pdf",
    "PDD":   "2016permenristekdikti061.pdf",
    "FKU":   "BUKU-PANDUAN-AKADEMIK-S1-KEDOKTERAN-FK-UNRI-3.pdf",
    "FISIP": "Files_2026_juknis-fisip-berdampak-untuk-negeri-fisip-unri_1782799324_6a435bdc3156e.pdf",
    "SMBT":  "PANDUAN-SMBT-UNRI-2026-v1.pdf",
    "PKKMB": "PANDUAN_PKKMB_2024_FINAL-30072024.pdf",
    "REG":   "Panduan-registrasi-ulang.pdf",
    "PGB":   "Peraturan-Rektor-Tentang-Pelaksanaan-Pengabdian-Universitas-Riau.pdf",
}

# Dokumen yang sengaja dibiarkan tanpa gold (lihat catatan di atas).
DISTRAKTOR = [
    "KEPUTUSAN REKTOR UNIVERSITAS RIAU NOMOR 1694 TAHUN 2025 TENTANG PENETAPAN PANDUAN PENELITIAN DAN PENGABDIAN KEPADA MASYARAKAT UNIVERSITAS RIAU TAHUN 2025...pdf",
    "Files_2024_panduan-kukerta-reguler-dan-mbkm-tahun-2023-10-maret-2023_1725259087_66d55d4f93bbf.pdf",
    "2016permenristekdikti005.pdf",
    "2017permenristekdikti039.pdf",
]

# Pertanyaan lama yang menjadi bermasalah setelah corpus diperluas, lalu
# ditulis ulang. Kunci = pertanyaan lama, nilai = pertanyaan pengganti.
UNANSWERABLE_BASI = {
    # Panduan PKKMB 2024 halaman 11 menyebut nama Dekan Fakultas Teknik,
    # sehingga label unanswerable-nya tidak lagi sah.
    "Siapa Dekan Fakultas Teknik UNRI sekarang?":
        "Siapa Dekan Fakultas Hukum Universitas Riau sekarang?",
    # Panduan PKKMB 2024 juga memuat jadwal PKKMB tahun itu, sehingga
    # pertanyaan tanpa keterangan tahun jadi punya dua jawaban benar.
    "Kapan pelaksanaan PKKMB mahasiswa baru?":
        "Kapan pelaksanaan PKKMB mahasiswa baru tahun akademik 2025/2026?",
}

# Lokasi gold TAMBAHAN yang muncul karena dokumen baru memuat aturan yang
# sama dan masih berlaku. Dipakai bersama gold_mode "any": cukup salah satu
# lokasi yang ditemukan. Kunci = teks pertanyaan (setelah penulisan ulang).
GOLD_EXTRA_BARU = {
    # Pasal 3 Permendikbudristek 6/2022 menyatakan hal yang persis sama
    # dengan Peraturan Rektor tentang penerbitan ijazah.
    "Apa status hukum ijazah dan transkrip akademik?":
        "2022permendikbudristek006.pdf:p4",
    # Aturan logbook harian dan portal pendaftaran DPL tidak berubah antara
    # Panduan Kukerta 2023 dan Panduan Kukerta Berdampak 2026, jadi kedua
    # lokasi sama-sama memuat jawaban yang berlaku.
    "Apa yang wajib saya isi setiap hari selama di lokasi kukerta?":
        "Files_2024_panduan-kukerta-reguler-dan-mbkm-tahun-2023-10-maret-2023_1725259087_66d55d4f93bbf.pdf:p13",
    "Bagaimana cara mendaftar jadi dosen pembimbing lapangan kukerta?":
        "Files_2024_panduan-kukerta-reguler-dan-mbkm-tahun-2023-10-maret-2023_1725259087_66d55d4f93bbf.pdf:p8",
}

# Pertanyaan yang gold_mode-nya menjadi "any" karena penambahan di atas.
GOLD_ANY_BARU = set(GOLD_EXTRA_BARU)


# --------------------------------------------------------------------------
# KOMPARATIF
# --------------------------------------------------------------------------
KOMPARATIF = [
    ("Kukerta gelombang 1 dan gelombang 2 bedanya apa?",
     "Keduanya sama-sama berlangsung 40 hari dan setara 4 SKS. Gelombang 1 dilaksanakan di wilayah Kota Pekanbaru pada rentang Februari hingga April 2026 dan diperuntukkan bagi mahasiswa yang tinggal menyelesaikan satu semester lagi atau menghadapi kondisi tertentu, sedangkan Gelombang 2 dilaksanakan di desa sasaran atau mitra sasaran.",
     "KKN:p15|KKN:p18", "komparatif", "hard", "yes"),

    ("Syarat jumlah SKS untuk ikut kukerta gelombang 1 dan gelombang 2 sama atau beda?",
     "Sama, yaitu telah selesai mengambil mata kuliah dengan beban minimal 100 SKS. Bedanya, Gelombang 1 menambah syarat bahwa mahasiswa tinggal menyelesaikan satu semester lagi pada semester ganjil atau menghadapi kondisi tertentu seperti kesehatan dan kondisi keluarga.",
     "KKN:p16|KKN:p18", "komparatif", "medium", "yes"),

    ("Kukerta reguler dan kukerta khusus bedanya di mana?",
     "Kukerta Berdampak Reguler dilaksanakan universitas di desa sasaran atau mitra sasaran, sedangkan Kukerta Berdampak Khusus diusulkan oleh Fakultas, berbasis keilmuan lintas program studi dalam satu fakultas, dan dilaksanakan pada desa binaan Fakultas.",
     "KKN:p18|KKN:p19", "komparatif", "hard", "yes"),

    ("Kukerta dan program FISIP Berdampak untuk Negeri dihargai berapa SKS?",
     "Kukerta Berdampak Reguler berlangsung 40 hari dan setara 4 SKS, sedangkan FISIP Berdampak untuk Negeri dikonversi menjadi 13 SKS dengan hitungan 10 minggu setara 364 jam.",
     "KKN:p15|FISIP:p10", "komparatif", "hard", "yes"),

    ("Syarat SKS untuk ikut kukerta dan untuk ikut FISIP Berdampak untuk Negeri sama?",
     "Berbeda. Peserta kukerta harus telah menyelesaikan mata kuliah dengan beban minimal 100 SKS, sedangkan peserta FISIP Berdampak untuk Negeri cukup telah menempuh minimal 80 SKS.",
     "KKN:p16|FISIP:p5", "komparatif", "medium", "yes"),

    ("Pembekalan umum dan pembekalan khusus kukerta bedanya apa?",
     "Pembekalan umum dilakukan universitas dengan penyaji dari dalam dan luar universitas, sedangkan pembekalan khusus dilakukan secara tutorial oleh Dosen Pembimbing Lapangan.",
     "KKN:p11", "komparatif", "easy", "yes"),

    ("Program wajib dan program tambahan dalam kukerta bedanya apa?",
     "Program wajib diselenggarakan bersama-sama oleh kelompok sesuai Rencana Kerja yang disetujui DPL dan secara kuantitatif memenuhi minimal 480 jam setara 15 minggu, sedangkan program tambahan muncul setelah matriks perencanaan disahkan untuk mengakomodasi kebutuhan masyarakat yang belum tercantum.",
     "KKN:p8", "komparatif", "medium", "yes"),

    ("Kukerta yang diselenggarakan pihak luar kampus nilainya dihitung sama?",
     "Ya. Program Kuliah Kerja Nyata eksternal disetarakan dengan mata kuliah Kuliah Kerja Nyata sehingga nilai yang didapat mahasiswa dapat menjadi nilai mata kuliah Kuliah Kerja Nyata.",
     "KKN:p22", "komparatif", "medium", "yes"),

    ("Syarat ikut kukerta dari lembaga luar lebih berat dibanding kukerta biasa?",
     "Ya. Kukerta eksternal mensyaratkan minimal 100 SKS ditambah Indeks Prestasi Kumulatif paling rendah 3.0, pernah atau sedang aktif sebagai pengurus atau anggota organisasi kemahasiswaan, surat keterangan sehat dan berkelakuan baik, serta tidak sedang hamil, sedangkan kukerta reguler hanya mensyaratkan 100 SKS, status mahasiswa aktif, dan lulus pembekalan.",
     "KKN:p18|KKN:p23", "komparatif", "hard", "yes"),

    ("Bobot nilai dari masyarakat dan dari luaran kukerta, mana yang lebih besar?",
     "Luaran lebih besar. Penilaian masyarakat berbobot 20 persen, sedangkan luaran berbobot 40 persen.",
     "KKN:p17|KKN:p18", "komparatif", "medium", "yes"),

    ("Skema Riset Afirmasi dan Riset Peningkatan Kapasitas Dosen Muda bedanya apa?",
     "Riset Afirmasi berdana maksimal Rp10.000.000 dengan luaran wajib artikel di jurnal bereputasi nasional Sinta 1-5 dan diprioritaskan bagi dosen yang belum lulus pendanaan proposal sumber dana DIPA Universitas Riau. Riset Peningkatan Kapasitas Dosen Muda berdana maksimal Rp15.000.000 dengan luaran wajib jurnal nasional terindeks minimal Sinta 3 atau prosiding internasional bereputasi.",
     "PKM:p15|PKM:p16", "komparatif", "hard", "yes"),

    ("Hibah riset mana yang dananya paling besar di Universitas Riau?",
     "Riset Inovasi dan Hilirisasi Industri, dengan besaran dana maksimal Rp80.000.000.",
     "PKM:p21|PKM:p52", "komparatif", "easy", "yes"),

    ("Syarat ketua peneliti riset kolaborasi internasional dan riset unggulan bedanya apa?",
     "Ketua Riset Kolaborasi Internasional harus Doktor dengan jabatan akademik minimal Lektor, bukan CPNS, dan memiliki SINTA Score Overall minimal 150 untuk bidang saintek atau 50 untuk soshum dan seni. Ketua Riset Unggulan Universitas Riau berjabatan akademik minimal asisten ahli untuk S3 atau lektor untuk S2, bukan CPNS, dengan SINTA Score Overall minimal 200 untuk saintek atau 100 untuk soshum dan seni.",
     "PKM:p17|PKM:p18", "komparatif", "hard", "yes"),

    ("Riset Produk dan Prototipe dan Riset Inovasi dan Hilirisasi Industri bedanya di mana?",
     "Riset Produk dan Prototipe berada pada Tingkat Kesiapterapan Teknologi level 4-6 dengan dana maksimal Rp60.000.000, sedangkan Riset Inovasi dan Hilirisasi Industri berada pada level 6-9 dengan dana maksimal Rp80.000.000 dan wajib memiliki mitra dengan dukungan pendanaan in cash atau in kind minimal 10 persen.",
     "PKM:p19|PKM:p21", "komparatif", "hard", "yes"),

    ("Skema pengabdian Pemberdayaan Berbasis Masyarakat dan Pemberdayaan Desa Binaan bedanya apa?",
     "Pemberdayaan Berbasis Masyarakat berdana maksimal Rp30.000.000 dan bersifat mono tahun, sedangkan Pemberdayaan Desa Binaan berdana maksimal Rp40.000.000 dan bersifat multi tahun.",
     "PKM:p52", "komparatif", "medium", "yes"),

    ("Berapa dosen minimal di tim Pemberdayaan Berbasis Masyarakat dibanding tim Pemberdayaan Desa Binaan?",
     "Anggota pengusul Pemberdayaan Berbasis Masyarakat minimal 3 orang dosen, sedangkan anggota pengusul Pemberdayaan Desa Binaan minimal 5 orang dosen.",
     "PKM:p23|PKM:p26", "komparatif", "medium", "yes"),

    ("Kunjungan ke lokasi mitra minimal berapa kali untuk skema berbasis masyarakat dan skema desa binaan?",
     "Skema Pemberdayaan Berbasis Masyarakat sekurang-kurangnya 3 kali kunjungan, sedangkan Pemberdayaan Desa Binaan sekurang-kurangnya 4 kali kunjungan.",
     "PKM:p23|PKM:p26", "komparatif", "medium", "yes"),

    ("Dosen tetap dan dosen tidak tetap bedanya apa?",
     "Dosen tetap bekerja penuh waktu pada perguruan tinggi dan memenuhi beban kerja paling sedikit sepadan dengan 12 satuan kredit semester, sedangkan dosen tidak tetap tidak bekerja penuh waktu dan/atau memenuhi beban kerja kurang dari 12 satuan kredit semester.",
     "PMD44:p2", "komparatif", "easy", "yes"),

    ("Promosi dan demosi dosen bedanya apa?",
     "Promosi merupakan kenaikan jenjang jabatan akademik satu jenjang lebih tinggi, sedangkan demosi merupakan penurunan jenjang jabatan akademik satu jenjang lebih rendah.",
     "PMD44:p12|PMD44:p13", "komparatif", "medium", "yes"),

    ("Tugas Lektor Kepala dan Profesor bedanya di mana?",
     "Lektor Kepala memiliki kepakaran dalam bidang ilmu tertentu, mengembangkan keilmuan sesuai kepakarannya, dan membina Asisten Ahli dan/atau Lektor di bawah pembinaan Profesor. Profesor memiliki kepakaran, otoritas, dan wibawa ilmiah dalam bidang ilmunya, memimpin pengembangan keilmuan sesuai kepakarannya, serta membina Asisten Ahli, Lektor, dan/atau Lektor Kepala.",
     "PMD44:p3", "komparatif", "medium", "yes"),

    ("Syarat pendidikan dosen untuk mengajar program sarjana dan program magister sama?",
     "Berbeda. Untuk program diploma atau program sarjana, kualifikasi akademik dosen paling rendah magister atau magister terapan, sedangkan untuk program magister, magister terapan, doktor, atau doktor terapan paling rendah doktor atau doktor terapan.",
     "PMD44:p4", "komparatif", "medium", "yes"),

    ("Tunjangan profesi dan tunjangan kehormatan dosen besarnya sama?",
     "Tidak. Tunjangan profesi setara dengan 1 kali gaji pokok dosen aparatur sipil negara, sedangkan tunjangan kehormatan bagi Profesor setara dengan 2 kali gaji pokok dosen aparatur sipil negara.",
     "PMD44:p21", "komparatif", "medium", "yes"),

    ("Batas usia pensiun dosen biasa dan profesor sama?",
     "Tidak sama. Batas usia pensiun 65 tahun, sedangkan bagi dosen dengan jabatan akademik Profesor 70 tahun.",
     "PMD44:p20", "komparatif", "easy", "yes"),

    ("Bidang yang ditangani wakil rektor dan wakil dekan di UNRI sama saja?",
     "Tidak sama. Wakil Rektor terdiri atas Bidang Akademik, Bidang Keuangan dan Umum, Bidang Kemahasiswaan dan Alumni, serta Bidang Perencanaan, Kerja Sama, dan Sistem Informasi. Wakil Dekan terdiri atas Bidang Akademik, Bidang Keuangan dan Umum, serta Bidang Kemahasiswaan, Alumni, dan Kerja Sama.",
     "PDS16:p3|PDS16:p5", "komparatif", "hard", "yes"),

    ("Tugas lembaga penelitian dan lembaga penjaminan mutu di UNRI bedanya apa?",
     "Lembaga Penelitian dan Pengabdian Kepada Masyarakat melaksanakan koordinasi, pelaksanaan, pemantauan, dan evaluasi kegiatan penelitian dan pengabdian kepada masyarakat, sedangkan Lembaga Penjaminan Mutu dan Pengembangan Pembelajaran melaksanakan koordinasi, pelaksanaan, pemantauan, dan evaluasi kegiatan penjaminan mutu dan pengembangan pembelajaran.",
     "PDS16:p11|PDS16:p13", "komparatif", "medium", "yes"),

    ("Susunan organisasi Fakultas Pertanian berbeda dari fakultas lain di UNRI?",
     "Ya. Susunan organisasi Fakultas Pertanian memuat laboratorium, bengkel, studio, dan kebun percobaan, sedangkan fakultas lainnya hanya laboratorium, bengkel, dan studio.",
     "PDS16:p5", "komparatif", "hard", "yes"),

    ("Ketua jurusan dan koordinator program studi bedanya apa?",
     "Ketua jurusan bertanggung jawab kepada dekan dan bertugas memimpin serta melaksanakan penyelenggaraan jurusan berdasarkan kebijakan dekan, sedangkan koordinator Program Studi adalah dosen yang ditunjuk Rektor dan bertanggung jawab kepada ketua jurusan.",
     "PDS16:p6", "komparatif", "medium", "yes"),

    ("Jalur SMBT dan jalur PBM di UNRI bedanya apa?",
     "SMBT menjaring lulusan SMA, SMK, dan MA yang unggul melalui Ujian Tulis Berbasis Komputer, sedangkan PBM menjaring lulusan yang berprestasi di bidang olahraga, seni, sains, dan hafiz Al-Qur'an, dengan hasil UTBK hanya menjadi pertimbangan bila peminat melebihi daya tampung.",
     "SMBT:p1", "komparatif", "medium", "yes"),

    ("Jalur afirmasi dan jalur kerja sama UNRI diperuntukkan bagi siapa?",
     "Jalur Afirmasi diperuntukkan bagi anak kandung dosen atau tenaga kependidikan berstatus ASN atau P3K di lingkungan Universitas Riau, sedangkan jalur Kerja Sama diperuntukkan bagi calon mahasiswa program studi D3 Teknologi Pulp dan Kertas hasil kerja sama dengan PT RAPP dan harus memiliki rekomendasi dari PT RAPP.",
     "SMBT:p1|SMBT:p5", "komparatif", "hard", "yes"),

    ("Pilihan program studi untuk pendaftar dari Riau dan dari luar Riau sama?",
     "Tidak sama. Pendaftar yang asal sekolahnya dari Provinsi Riau dapat memilih sebanyak-banyaknya 2 program studi dari seluruh program studi yang ditawarkan, sedangkan pendaftar dari luar Provinsi Riau hanya dapat memilih paling banyak 2 program studi yang terdaftar di Fakultas Perikanan dan Kelautan.",
     "SMBT:p12", "komparatif", "medium", "yes"),

    ("Ijazah, sertifikat kompetensi, dan sertifikat profesi bedanya apa?",
     "Ijazah diberikan kepada lulusan pendidikan akademik dan pendidikan vokasi sebagai pengakuan terhadap prestasi belajar dan/atau penyelesaian program studi terakreditasi. Sertifikat kompetensi memuat pengakuan kompetensi atas prestasi lulusan sesuai keahlian dalam cabang ilmunya. Sertifikat profesi memuat pengakuan untuk melakukan praktik profesi yang diperoleh lulusan pendidikan profesi.",
     "IJZ22:p2", "komparatif", "medium", "yes"),

    ("Yang menandatangani ijazah di universitas dan di sekolah tinggi sama?",
     "Berbeda. Untuk universitas dan institut, ijazah ditandatangani oleh rektor dan dekan, sedangkan untuk sekolah tinggi oleh ketua dan pemimpin unit pengelola program studi.",
     "IJZ22:p5", "komparatif", "medium", "yes"),

    ("Gratifikasi yang wajib dilaporkan dan yang tidak wajib dilaporkan bedanya apa?",
     "Gratifikasi yang wajib dilaporkan adalah gratifikasi yang berhubungan dengan jabatan, yaitu dalam rangka mempengaruhi kebijakan, keputusan, atau perlakuan pemangku kewenangan termasuk yang memiliki benturan kepentingan, dalam rangka kunjungan dinas, dan dalam proses penerimaan, promosi, atau mutasi pejabat atau pegawai. Selain itu masuk kategori yang tidak wajib dilaporkan.",
     "GRAT:p4", "komparatif", "hard", "yes"),

    ("Batas nilai pemberian sesama pegawai dan sesama rekan kerja yang tidak wajib dilaporkan sama?",
     "Tidak sama. Pemberian sesama pegawai dalam rangka pisah sambut, pensiun, promosi jabatan, dan ulang tahun paling banyak Rp300.000 per pemberian per orang, sedangkan pemberian sesama rekan kerja paling banyak Rp200.000 per pemberi per orang. Keduanya dibatasi total Rp1.000.000 dalam satu tahun dari pemberi yang sama.",
     "GRAT:p5", "komparatif", "hard", "yes"),

    ("Biaya langsung dan biaya tidak langsung dalam standar biaya operasional pendidikan tinggi bedanya apa?",
     "Biaya langsung merupakan biaya operasional yang terkait langsung dengan penyelenggaraan program studi, sedangkan biaya tidak langsung merupakan biaya operasional pengelolaan institusi yang diperlukan dalam mendukung penyelenggaraan program studi.",
     "SSBO:p3", "komparatif", "medium", "yes"),

    ("Kegiatan inti dan kegiatan pendukung pada FISIP Berdampak untuk Negeri bedanya apa?",
     "Kegiatan inti adalah kegiatan besar yang diselenggarakan bersama-sama sesuai Laporan Rencana Kerja yang dibuat dan disetujui DPL dan dikonversi dengan mata kuliah 13 SKS, sedangkan kegiatan pendukung berkaitan dengan muatan mata kuliah konversi yang dilakukan tiap individu sesuai mata kuliah konversi yang diambil.",
     "FISIP:p9", "komparatif", "medium", "yes"),
]

# --------------------------------------------------------------------------
# MULTI-HOP - jawabannya harus dirangkai dari dua halaman atau dua dokumen
# --------------------------------------------------------------------------
MULTI_HOP = [
    ("Saya harus lulus berapa SKS untuk ikut kukerta, dan kukerta itu sendiri dihitung berapa SKS?",
     "Peserta harus telah selesai mengambil mata kuliah dengan beban minimal 100 SKS, sedangkan kukerta itu sendiri setara dengan 4 SKS.",
     "KKN:p15|KKN:p16", "multi-hop", "medium", "yes"),

    ("Kukerta bernilai berapa SKS dan berapa jam kerja minimal yang harus saya tempuh?",
     "Kukerta setara dengan 4 SKS. Jumlah total jam kerja yang harus ditempuh setiap mahasiswa peserta minimal 272 jam kerja selama pelaksanaan.",
     "KKN:p8|KKN:p15", "multi-hop", "medium", "yes"),

    ("Pembimbing lapangan kukerta harus punya NIDN, lalu apa syarat pendidikan minimal seorang dosen program sarjana?",
     "Dosen Pembimbing Lapangan harus memiliki Nomor Induk Dosen Nasional atau Nomor Unik Pendidik dan Tenaga Kependidikan serta berstatus dosen aktif Universitas Riau. Kualifikasi akademik dosen pada program diploma atau program sarjana paling rendah magister atau magister terapan.",
     "KKN:p13|PMD44:p4", "multi-hop", "hard", "yes"),

    ("Ketua penelitian di UNRI harus dosen tetap, lalu apa yang dimaksud dosen tetap?",
     "Ketua peneliti atau pelaksana pengabdi adalah dosen tetap Universitas Riau yang mempunyai NIDN, NUPTK, atau NIDK dan sudah mempunyai jabatan fungsional. Dosen tetap bekerja penuh waktu pada perguruan tinggi dan memenuhi beban kerja paling sedikit sepadan dengan 12 satuan kredit semester.",
     "PKM:p9|PMD44:p2", "multi-hop", "hard", "yes"),

    ("Ketua skema riset kolaborasi internasional minimal Lektor, lalu apa tugas seorang Lektor?",
     "Ketua Riset Kolaborasi Internasional harus Doktor dengan jabatan akademik minimal Lektor. Lektor melaksanakan Tridharma secara mandiri di bawah pembinaan Lektor Kepala dan/atau Profesor.",
     "PKM:p17|PMD44:p3", "multi-hop", "hard", "yes"),

    ("Lembaga mana yang mengelola kukerta di UNRI dan apa tugas resmi lembaga itu?",
     "Pengelolanya adalah Lembaga Penelitian dan Pengabdian kepada Masyarakat Universitas Riau, yang bertugas melaksanakan koordinasi, pelaksanaan, pemantauan, dan evaluasi kegiatan penelitian dan pengabdian kepada masyarakat.",
     "KKN:p11|PDS16:p11", "multi-hop", "hard", "yes"),

    ("Kukerta khusus diusulkan fakultas, lalu apa tugas fakultas dalam struktur organisasi UNRI?",
     "Kukerta Berdampak Khusus diusulkan oleh Fakultas dan dilaksanakan pada desa binaan Fakultas. Fakultas bertugas menyelenggarakan dan mengelola pendidikan akademik, pendidikan vokasi, dan/atau pendidikan profesi dalam satu atau beberapa pohon atau kelompok ilmu pengetahuan dan/atau teknologi.",
     "KKN:p19|PDS16:p4", "multi-hop", "hard", "yes"),

    ("Kukerta gelombang 1 bisa diusulkan Dekan ke Wakil Rektor Bidang Akademik, apa tugas jabatan itu?",
     "Pelaksanaan kukerta Gelombang 1 dapat dilaksanakan berdasarkan usulan Dekan kepada Wakil Rektor Bidang Akademik. Wakil Rektor Bidang Akademik bertugas membantu Rektor dalam memimpin penyelenggaraan kegiatan di bidang pendidikan, penelitian, dan pengabdian kepada masyarakat.",
     "KKN:p15|PDS16:p3", "multi-hop", "hard", "yes"),

    ("Naik jadi profesor butuh publikasi ilmiah, skema riset UNRI mana yang luaran wajibnya jurnal internasional bereputasi?",
     "Promosi ke jenjang jabatan akademik Profesor mensyaratkan antara lain memiliki publikasi ilmiah. Riset Unggulan Universitas Riau mensyaratkan luaran wajib berupa artikel di jurnal internasional terindeks pada basis data internasional bereputasi.",
     "PMD44:p12|PKM:p18", "multi-hop", "hard", "yes"),

    ("Berapa beban kerja minimal dosen tetap dan kegiatan apa saja yang dihitung di dalamnya?",
     "Beban kerja dosen tetap paling sedikit sepadan dengan 12 satuan kredit semester. Kegiatan pokoknya mencakup merencanakan pembelajaran, melaksanakan proses pembelajaran, melakukan evaluasi pembelajaran, membimbing dan melatih, melakukan penelitian, melakukan tugas tambahan, serta melakukan pengabdian kepada masyarakat.",
     "PMD44:p2|PMD44:p9", "multi-hop", "hard", "yes"),

    ("Poster hasil kukerta ukurannya berapa dan videonya maksimal berapa menit?",
     "Poster berukuran 60 cm x 160 cm posisi portrait, sedangkan video dokumenter berdurasi maksimal 5 menit.",
     "KKN:p24", "multi-hop", "medium", "yes"),

    ("Siapa yang menilai peserta kukerta khusus dan berapa bobot penilaian dari masyarakat?",
     "Penilaian peserta Kukerta Berdampak Khusus dilakukan oleh DPL Fakultas dan berkoordinasi dengan LPPM. Bobot penilaian masyarakat yang diinput oleh DPL adalah 20 persen.",
     "KKN:p20|KKN:p22", "multi-hop", "hard", "yes"),

    ("Kukerta gelombang 2 dilaksanakan bulan apa dan kapan nilainya dikirim ke fakultas?",
     "Pelaksanaan kukerta Gelombang 2 berlangsung Juni hingga Agustus 2026, sedangkan pengiriman nilai ke fakultas dijadwalkan September 2026.",
     "KKN:p17|KKN:p18", "multi-hop", "medium", "yes"),

    ("Untuk naik jadi Profesor perlu pengalaman berapa tahun, dan berapa profesor yang harus ada di tim promosinya?",
     "Diperlukan pengalaman kerja 10 tahun sebagai dosen tetap. Tim promosi dosen terdiri atas paling sedikit 5 Profesor pada rumpun ilmu bidang studi dosen yang dipromosikan, dengan paling sedikit 3 di antaranya Profesor dari perguruan tinggi lain.",
     "PMD44:p12|PMD44:p13", "multi-hop", "hard", "yes"),

    ("Kalau kampus mengangkat dosen yang tidak memenuhi kualifikasi, sanksinya apa dan apa kualifikasi minimal itu?",
     "Kualifikasi akademik dosen paling rendah magister atau magister terapan untuk program diploma atau sarjana dan doktor atau doktor terapan untuk program magister atau doktor. PTN badan hukum atau Badan Penyelenggara yang melanggar larangan tersebut dikenai sanksi administratif berupa peringatan tertulis.",
     "PMD44:p4|PMD44:p6", "multi-hop", "hard", "yes"),

    ("Apa saja luaran wajib kukerta dan ke mana video dokumenternya diunggah?",
     "Luaran kukerta meliputi poster yang didaftarkan hak cipta, hak cipta video dokumenter, submit dan draft artikel pengabdian pada jurnal ISSN atau prosiding Seminar Nasional Pengabdian kepada Masyarakat LPPM Unri, buku profil desa, serta publikasi media massa cetak atau elektronik. Video diunggah pada laman youtube LPPM.",
     "KKN:p24|KKN:p25", "multi-hop", "hard", "yes"),

    ("Ijazah diterbitkan bersama dokumen apa saja dan ditulis dalam bahasa apa?",
     "Ijazah diterbitkan perguruan tinggi disertai dengan Transkrip Akademik dan Surat Keterangan Pendamping Ijazah. Ketiganya ditulis dalam bahasa Indonesia dan dapat diterjemahkan dalam bahasa Inggris atau bahasa asing lainnya.",
     "IJZ22:p4|IJZ22:p5", "multi-hop", "hard", "yes"),

    ("Mahasiswa FISIP Berdampak untuk Negeri dinilai oleh siapa saja dan berapa bobot laporan akhirnya?",
     "Penilaian dilakukan oleh Dosen Pembimbing Lapangan dan Dosen Pengampu Mata Kuliah. Laporan Akhir berbobot 15 persen dan dinilai oleh Dosen Pembimbing Lapangan.",
     "FISIP:p6|FISIP:p8", "multi-hop", "hard", "yes"),

    ("Program FISIP Berdampak untuk Negeri dikonversi jadi berapa SKS dan apa saja luarannya?",
     "Dikonversi menjadi 13 SKS. Luarannya meliputi laporan kegiatan pengabdian masyarakat, buku profil desa, bukti submit pengabdian kepada masyarakat pada jurnal atau prosiding atau book chapter SNPM FISIP Universitas Riau, publikasi media massa atau elektronik, hak cipta poster, dan hak cipta video dokumenter.",
     "FISIP:p10|FISIP:p11", "multi-hop", "hard", "yes"),

    ("Kapan pendaftaran jalur mandiri UNRI dibuka dan berapa daya tampung maksimalnya?",
     "Pendaftaran dimulai 4 Mei 2026 sampai 4 Juli 2026 pukul 15.00 WIB. Jumlah alokasi daya tampung mahasiswa baru jalur mandiri paling banyak 30 persen pada setiap program studi.",
     "SMBT:p2|SMBT:p12", "multi-hop", "hard", "yes"),

    ("Daftar ulang mahasiswa baru UNRI ada berapa tahap dan dokumen apa yang dipakai memvalidasi UKT?",
     "Ada 5 tahap, yaitu update biodata dan upload dokumen persyaratan, verifikasi atau validasi dokumen, pembayaran UKT beserta biaya tes narkoba, tes kesehatan, dan buta warna, generate NIM, dan cetak e-KTM. Dokumen syarat validasi UKT meliputi slip gaji atau surat keterangan penghasilan orang tua, kartu keluarga, bukti bayar listrik, kartu KPS, bukti pajak motor, bukti pajak mobil, dan bukti kepemilikan lahan.",
     "REG:p1|REG:p2", "multi-hop", "hard", "yes"),

    ("Berapa masa studi maksimal dan beban studi mahasiswa kedokteran UNRI, dan berapa total SKS di tabel kurikulumnya?",
     "Masa studi maksimal 12 semester dengan beban studi 148 SKS, sedangkan tabel daftar mata kuliah per semester mencantumkan total 151 SKS.",
     "FKU:p24|FKU:p29", "multi-hop", "hard", "yes"),

    ("Kapan mahasiswa kedokteran UNRI boleh menyusun proposal skripsi dan di semester berapa skripsi dijadwalkan?",
     "Mahasiswa diperkenankan menyusun proposal skripsi jika telah lulus 75 SKS dan telah lulus blok metode penelitian. Skripsi dijadwalkan pada semester VIII.",
     "FKU:p25|FKU:p46", "multi-hop", "hard", "yes"),

    ("Berapa persen kehadiran minimal kuliah pakar di FK UNRI dan apa akibatnya kalau nilai modul di bawah batas?",
     "Kehadiran pada kegiatan kuliah pakar minimal 80 persen. Mahasiswa dengan nilai rata-rata modul kurang dari 55 wajib mengikuti remedial.",
     "FKU:p37|FKU:p41", "multi-hop", "hard", "yes"),

    ("Gratifikasi yang tidak bisa ditolak harus diapakan dan siapa yang menerima laporannya?",
     "Pegawai wajib menolak gratifikasi yang berhubungan dengan jabatan, dan bila tidak dapat ditolak wajib melaporkannya kepada satuan tugas pengendalian gratifikasi. Unit pengendalian gratifikasi menerima dan memproses laporan gratifikasi kedinasan serta meneruskan laporan yang bukan kedinasan kepada Komisi Pemberantasan Korupsi.",
     "GRAT:p6|GRAT:p7", "multi-hop", "hard", "yes"),

    ("Apa tujuan pangkalan data pendidikan tinggi dan data apa saja yang dihimpunnya?",
     "Pangkalan Data Pendidikan Tinggi bertujuan antara lain mewujudkan basis data tunggal dalam perencanaan, pengaturan, pembinaan, dan pengawasan pendidikan tinggi. Data yang dihimpun terdiri atas Data Pokok Pendidikan Tinggi, Data Referensi Pendidikan Tinggi, dan Data Transaksional Pendidikan Tinggi.",
     "PDD:p3|PDD:p4", "multi-hop", "hard", "yes"),

    ("Tahapan pengelolaan pengabdian di UNRI apa saja dan lewat sistem apa pengusulannya?",
     "Tahapan kegiatan pengabdian meliputi pengumuman, pengusulan, penyeleksian atau penunjukan, penetapan, pelaksanaan, pengawasan, pelaporan, dan penilaian luaran. Semua proses dilakukan melalui Sistem Elektronik Penelitian dan Pengabdian atau e-ppm.",
     "PGB:p4|PKM:p5", "multi-hop", "hard", "yes"),

    ("Peserta PKKMB UNRI 2024 siapa saja dan berapa hari kegiatannya?",
     "Pesertanya adalah mahasiswa baru jenjang D3, D4, dan S1 dari 10 fakultas di Universitas Riau yang diterima pada Tahun Ajaran 2024/2025. Kegiatan berlangsung tiga hari, yaitu 6 sampai 8 Agustus 2024.",
     "PKKMB:p7", "multi-hop", "medium", "yes"),
]

# --------------------------------------------------------------------------
# PROSEDUR
# --------------------------------------------------------------------------
PROSEDUR = [
    ("Bagaimana alur mendaftar jadi peserta kukerta?",
     "Calon peserta mendaftar pembekalan melalui portal KUKERTA yang disediakan LPPM menggunakan email institusi, mengikuti pembekalan, lalu bagi yang dinyatakan lulus pembekalan dapat melanjutkan ke tahap pendaftaran melalui portal KUKERTA.",
     "KKN:p11|KKN:p12", "prosedur", "hard", "yes"),

    ("Bagaimana cara mendaftar jadi dosen pembimbing lapangan kukerta?",
     "Mengisi formulir pendaftaran sebagai calon DPL secara online pada portal KUKERTA di https://kukerta.Unri.ac.id atau media yang disediakan LPPM, kemudian mengikuti pertemuan persamaan persepsi dan pembekalan program KUKERTA.",
     "KKN:p14", "prosedur", "medium", "yes"),

    ("Kalau saya tidak lulus pembekalan kukerta, apa yang harus dilakukan?",
     "Mahasiswa peserta yang dinyatakan tidak lulus wajib mengulang proses dari awal.",
     "KKN:p12", "prosedur", "easy", "yes"),

    ("Bagaimana matriks program kerja kukerta disusun sampai disahkan?",
     "Matriks program kerja yang sudah tersusun masuk pada tahap verifikasi, validasi, dan persetujuan oleh Dosen Pembimbing Lapangan, penanggung jawab lokasi, dan mahasiswa penyusun matriks. Matriks harus selesai maksimal 7 hari setelah di lokasi, diawali tahap observasi, sosialisasi program ke tokoh masyarakat, dan konsultasi dengan DPL.",
     "KKN:p8", "prosedur", "hard", "yes"),

    ("Bagaimana nilai kukerta akhirnya bisa masuk ke portal akademik?",
     "Laporan dan luaran mahasiswa dikoreksi DPL dan diberikan masa revisi, lalu DPL menginput nilai melalui portal KUKERTA. Setelah LPPM memvalidasi laporan, luaran, dan nilai, hasilnya dikirim ke Fakultas masing-masing untuk diinput ke dalam Portal Akademis.",
     "KKN:p12", "prosedur", "hard", "yes"),

    ("Kalau ada masalah di lokasi kukerta yang tidak bisa diselesaikan pembimbing, harus bagaimana?",
     "DPL berkonsultasi dan berkoordinasi dengan koordinator lokasi. Selanjutnya koordinator lokasi menyampaikan kepada Koordinator Layanan KUKERTA dan Kepala LPPM Unri untuk dicarikan solusinya.",
     "KKN:p13", "prosedur", "medium", "yes"),

    ("Bagaimana tahapan pelaksanaan kukerta khusus?",
     "Tahap 1 meliputi pendaftaran, pembekalan oleh LPPM, pendampingan kelompok oleh DPL, dan pemaparan program kerja kepada DPL. Tahap 2 adalah implementasi program di lokasi. Tahap 3 adalah penutupan, lokakarya, dan pelaporan.",
     "KKN:p20", "prosedur", "medium", "yes"),

    ("Apa yang wajib saya isi setiap hari selama di lokasi kukerta?",
     "Mahasiswa wajib mengisi logbook harian di portal KUKERTA secara realtime.",
     "KKN:p12", "prosedur", "easy", "yes"),

    ("Bagaimana cara mendaftarkan hak cipta poster hasil kukerta?",
     "Poster didaftarkan hak cipta melalui LPPM, dengan ketentuan ukuran 60 cm x 160 cm posisi portrait, wajib mencantumkan logo Kemendiktisaintek dan logo perguruan tinggi, bersifat original, bebas plagiarisme, politik, dan SARA, memuat judul, tim pelaksana, instansi pemberi dana, resume dan hasil pelaksanaan kegiatan serta teknologi yang diterapkan, dan pada pelaporan softfile menggunakan jenis warna RGB.",
     "KKN:p24", "prosedur", "hard", "yes"),

    ("Sebelum menjatuhkan sanksi ke mahasiswa kukerta, apa yang dilakukan lebih dulu?",
     "Penetapan sanksi dilakukan oleh LPPM Unri setelah melalui pengkajian, pembahasan, dan musyawarah, dengan mempertimbangkan masukan dari DPL, penanggung jawab lokasi, maupun mahasiswa yang terlibat, serta melakukan penyelidikan dan pengumpulan bukti dan fakta dari tempat kejadian secara transparan dan akuntabel.",
     "KKN:p26", "prosedur", "hard", "yes"),

    ("Bagaimana tahapan pengelolaan penelitian dan pengabdian di UNRI?",
     "Tahapannya meliputi pengumuman, pengusulan, penyeleksian, penetapan, pelaksanaan, monitoring dan evaluasi, pelaporan, dan penilaian keluaran, seluruhnya melalui Sistem Elektronik Penelitian dan Pengabdian atau e-ppm.",
     "PKM:p5", "prosedur", "medium", "yes"),

    ("Apa saja yang harus diunggah peneliti saat melaporkan hasil penelitiannya?",
     "Peneliti dan pengabdi wajib mengunggah logbook atau reporting, laporan penggunaan dana berupa softcopy, Surat Pernyataan Tanggungjawab Belanja, Berita Acara Serah Terima Barang, softcopy laporan hasil, hardcopy laporan hasil rangkap satu, dan luaran yang dijanjikan ke laman e-ppm, serta menyerahkan salinan berkas hardcopy ke LPPM.",
     "PKM:p14", "prosedur", "hard", "yes"),

    ("Bagaimana cara mengusulkan riset kolaborasi dengan kampus luar negeri?",
     "Usulan harus relevan dengan kepakaran yang ditekuni dan lulus seleksi, melibatkan peneliti mitra dari perguruan tinggi luar negeri dengan minimal h-index Scopus 5, proposal ditulis dalam bahasa Inggris, kolaborasi dengan perguruan tinggi QS300, serta wajib melampirkan surat persetujuan bermitra dari perguruan tinggi luar negeri tersebut.",
     "PKM:p17", "prosedur", "hard", "yes"),

    ("Bagaimana kampus yang belum punya tim promosi bisa menaikkan dosennya jadi profesor?",
     "Perguruan tinggi yang belum memenuhi persyaratan dapat melakukan promosi ke jenjang Profesor setelah mendapatkan rekomendasi dari tim promosi dosen perguruan tinggi lain yang memenuhi persyaratan, atau dari tim promosi dosen yang dibentuk oleh Kementerian.",
     "PMD44:p13", "prosedur", "hard", "yes"),

    ("Bagaimana sertifikat pendidik dosen bisa dibatalkan?",
     "Kementerian memerintahkan secara tertulis kepada pemimpin perguruan tinggi yang menyelenggarakan sertifikasi dosen untuk membatalkan sertifikat pendidik. Apabila dalam jangka waktu 3 bulan perguruan tinggi tidak melaksanakan perintah tertulis tersebut, pembatalan dilakukan oleh Kementerian.",
     "PMD44:p9", "prosedur", "hard", "yes"),

    ("Kalau dosen pindah ke kampus lain, jabatan akademiknya bagaimana?",
     "Jabatan akademik dosen pada perguruan tinggi tujuan ditetapkan oleh perguruan tinggi tujuan sesuai dengan kualifikasi, kompetensi, dan prestasi dosen.",
     "PMD44:p3", "prosedur", "medium", "yes"),

    ("Bagaimana cara membentuk jurusan baru di UNRI?",
     "Pembentukan, perubahan, dan penutupan jurusan ditetapkan oleh Rektor setelah mendapat persetujuan dari direktur jenderal yang menyelenggarakan tugas di bidang pendidikan tinggi.",
     "PDS16:p6", "prosedur", "medium", "yes"),

    ("Bagaimana pusat di bawah lembaga penelitian UNRI dibentuk dan ditutup?",
     "Pembentukan dan penutupan pusat dilakukan oleh Rektor sesuai dengan kebutuhan, dan Rektor dapat menunjuk dosen atau pejabat fungsional lainnya sebagai kepala pusat.",
     "PDS16:p12", "prosedur", "medium", "yes"),

    ("Bagaimana koordinator program studi ditunjuk di UNRI?",
     "Dalam penyelenggaraan Program Studi pada jurusan, Rektor dapat menunjuk seorang dosen sebagai koordinator Program Studi yang bertanggung jawab kepada ketua jurusan.",
     "PDS16:p6", "prosedur", "easy", "yes"),

    ("Bagaimana cara mendaftar jalur mandiri UNRI sampai dapat kartu ujian?",
     "Peserta mendaftar online melalui laman https://pmb.unri.ac.id, membuat akun dan mengisi data awal, memilih jalur masuk, membayar melalui Virtual Account, melengkapi data diri, lalu melakukan finalisasi. Setelah finalisasi akan tampil informasi jadwal, sesi, dan lokasi ujian serta menu mengunduh kartu peserta ujian, dan pendaftaran dinyatakan selesai jika calon peserta sudah memiliki kartu peserta ujian.",
     "SMBT:p3|SMBT:p11", "prosedur", "hard", "yes"),

    ("Bagaimana cara membayar biaya pendaftaran jalur mandiri UNRI?",
     "Membayar uang pendaftaran sebesar Rp350.000 melalui bank mitra Unri yaitu BNI, BSI, BTN syariah, BRK, dan Mandiri, menggunakan Virtual Account yang dibuat melalui aplikasi. Pembayaran bisa dilakukan melalui ATM, internet banking, mobile banking, maupun teller di seluruh cabang mitra, dan kwitansi disimpan sebagai bukti bayar.",
     "SMBT:p12", "prosedur", "hard", "yes"),

    ("Bagaimana cara mengunggah portofolio untuk pendaftaran jalur mandiri UNRI?",
     "Portofolio berbentuk video diunggah ke laman Youtube dan URL-nya dimasukkan pada menu upload portofolio di aplikasi pendaftaran melalui https://pmb.unri.ac.id. Portofolio berupa gambar disimpan pada Google Drive dan tautannya dimasukkan pada menu upload, dengan memastikan tautan dapat diakses oleh reviewer.",
     "SMBT:p13", "prosedur", "medium", "yes"),

    ("Urutan tahapan daftar ulang mahasiswa baru UNRI seperti apa?",
     "Pendaftaran ulang dibagi atas 5 tahapan yang harus dilakukan berurutan, yaitu update biodata dan upload dokumen persyaratan, verifikasi atau validasi dokumen, pembayaran UKT dan biaya pemeriksaan tes narkoba, tes kesehatan, dan buta warna, generate NIM, lalu cetak e-KTM.",
     "REG:p1", "prosedur", "medium", "yes"),

    ("Kalau dokumen daftar ulang saya dinyatakan tidak valid, harus bagaimana?",
     "Jika ada tombol perbaiki, klik tombol perbaiki untuk memperbaiki dokumen yang tidak valid, perhatikan kolom keterangan untuk data yang tidak valid, lalu unggah ulang dokumen yang tidak valid tersebut.",
     "REG:p9", "prosedur", "medium", "yes"),

    ("Bagaimana cara login ke aplikasi daftar ulang mahasiswa baru UNRI?",
     "Login di https://registrasiulang.unri.ac.id menggunakan nomor pendaftaran yang ada pada kartu peserta, dengan password berupa gabungan nomor pendaftaran dan tahun lahir, lalu memasukkan hasil penjumlahan captcha yang ditampilkan.",
     "REG:p4", "prosedur", "medium", "yes"),

    ("Kalau ijazah saya hilang, bagaimana cara mengurus penggantinya?",
     "Perguruan tinggi dapat menerbitkan Surat Keterangan Pengganti yang dinilai sama dengan ijazah, atas permintaan pemilik ijazah, apabila ijazah rusak, hilang, atau musnah yang dibuktikan dengan keterangan tertulis dari kepolisian.",
     "IJZ22:p11", "prosedur", "medium", "yes"),

    ("Bagaimana cara mengurus penyetaraan ijazah dari kampus luar negeri?",
     "Penyetaraan ijazah dilakukan secara elektronik melalui sistem yang dikelola oleh Kementerian, dengan syarat perguruan tinggi atau program studi tersebut telah terakreditasi atau diakui pemerintah negara setempat, serta pengusul memiliki ijazah asli dan transkrip nilai asli beserta ijazah asli jenjang pendidikan sebelumnya.",
     "IJZ22:p14", "prosedur", "hard", "yes"),

    ("Bagaimana cara mengesahkan fotokopi ijazah kalau kampus penerbitnya sudah tutup?",
     "Pengesahan fotokopi dilakukan oleh Kepala Lembaga Layanan Pendidikan Tinggi, yang dapat mendelegasikan wewenangnya kepada pejabat lain di bawahnya. Jika perguruan tinggi telah berubah, pengesahan dilakukan oleh pemimpin perguruan tinggi baru hasil perubahan.",
     "IJZ22:p13", "prosedur", "hard", "yes"),
]

# --------------------------------------------------------------------------
# UNANSWERABLE - pertanyaan wajar, tetapi jawabannya memang tidak ada di corpus
#
# Setiap kandidat sudah diuji dengan grep ke seluruh 31 dokumen corpus.
# Yang ternyata terjawab (misalnya honor, nama dekan FK, golongan gaji pokok)
# dipindahkan menjadi pertanyaan biasa, bukan dibiarkan di sini.
# --------------------------------------------------------------------------
UNANSWERABLE = [
    "Berapa jumlah kelompok kukerta yang diberangkatkan UNRI tahun 2026?",
    "Desa mana saja yang jadi lokasi kukerta UNRI tahun 2026?",
    "Berapa uang saku yang diterima mahasiswa selama kukerta?",
    "Berapa honor yang diterima dosen pembimbing lapangan kukerta?",
    "Berapa kuota peserta kukerta gelombang 1?",
    "Apa sanksi bagi pembimbing lapangan yang terlambat mengunggah nilai kukerta?",
    "Apakah mahasiswa warga negara asing boleh ikut kukerta di UNRI?",
    "Berapa jumlah proposal penelitian yang didanai LPPM UNRI tahun 2026?",
    "Berapa total anggaran DIPA LPPM Universitas Riau tahun 2026?",
    "Berapa lama dana penelitian cair setelah kontrak ditandatangani?",
    "Siapa saja yang menjadi reviewer proposal penelitian di LPPM UNRI?",
    "Berapa jumlah desa binaan Universitas Riau?",
    "Berapa jumlah pusat yang ada di bawah LPPM Universitas Riau?",
    "Berapa jumlah paten yang dihasilkan Universitas Riau tahun 2025?",
    "Berapa jumlah dosen Universitas Riau yang sudah bergelar profesor?",
    "Berapa jumlah mahasiswa aktif Universitas Riau?",
    "Berapa jumlah tenaga kependidikan di Universitas Riau?",
    "Berapa jumlah program studi di Universitas Riau?",
    "Berapa lama masa jabatan Rektor Universitas Riau?",
    "Berapa jumlah koleksi buku perpustakaan Universitas Riau?",
    "Berapa daya tampung jalur SMBT untuk Fakultas Teknik?",
    "Berapa nilai minimal UTBK supaya lulus jalur SMBT UNRI?",
    "Berapa jumlah mahasiswa baru yang diterima UNRI tahun 2026?",
    "Berapa tarif UKT untuk mahasiswa baru Universitas Riau?",
    "Berapa biaya tes kesehatan dan tes narkoba bagi mahasiswa baru UNRI?",
    "Berapa denda keterlambatan pembayaran UKT di Universitas Riau?",
    "Berapa biaya kuliah program studi Kedokteran di UNRI?",
    "Kapan jadwal PKKMB Universitas Riau tahun 2026?",
]

# --------------------------------------------------------------------------
# LIST, DEFINISI, FACTOID - cakupan untuk dokumen baru
# --------------------------------------------------------------------------
LAINNYA = [
    # ---- list ----
    ("Dokumen apa saja yang harus disiapkan untuk daftar ulang mahasiswa baru UNRI?",
     "Dokumen syarat validasi UKT meliputi slip gaji atau surat keterangan penghasilan orang tua, kartu keluarga, bukti bayar listrik, kartu KPS, bukti pajak motor, bukti pajak mobil, dan bukti kepemilikan lahan. Dokumen syarat validasi pendaftaran ulang meliputi kartu peserta, pas foto, ijazah atau surat keterangan lulus, rapor sampai semester 5, surat pernyataan kesediaan tes narkoba, pemeriksaan kesehatan, dan tes buta warna, surat pernyataan kebenaran data, surat keterangan tanggungan orang tua, KTP orang tua, dan surat pernyataan pemindahan fasilitas kesehatan BPJS.",
     "REG:p2|REG:p3", "list", "hard", "yes"),

    ("Program studi apa saja yang mewajibkan tes buta warna saat daftar ulang di UNRI?",
     "Kedokteran, kedokteran hewan, keperawatan, teknik kimia, kimia, fisika, biologi, dan sistem informasi.",
     "REG:p3", "list", "medium", "yes"),

    ("Apa saja syarat mendaftar jalur SMBT di UNRI?",
     "Warga Negara Indonesia, lulusan SMA, SMK, MA atau sederajat tahun 2024, 2025, dan 2026, tidak menerima lulusan Paket C, tidak diterima melalui jalur SNBP dan SNBT atau diterima lewat SNBT namun belum daftar ulang, telah memiliki ijazah atau surat keterangan lulus yang memuat jati diri, pas foto, dan cap yang sah, serta dalam kondisi sehat sehingga tidak mengganggu proses UTBK.",
     "SMBT:p3", "list", "hard", "yes"),

    ("Apa saja yang dilarang selama mengikuti kukerta?",
     "Antara lain memberikan laporan yang merugikan pejabat atau lembaga mitra, memberikan informasi ke media yang menimbulkan opini negatif, membuat cap atau stempel kukerta, menggunakan tanda tangan tanpa izin, menghubungi instansi lebih tinggi tanpa izin, tinggal bersama lawan jenis dalam satu rumah tanpa keluarga setempat, menerima tamu bermalam, membawa kendaraan ke lokasi, mengemudi tanpa SIM, mengajukan proposal bantuan tanpa persetujuan DPL, mencemarkan nama baik universitas, melanggar peraturan perundang-undangan, serta meninggalkan lokasi sebelum berakhir atau tanpa izin DPL.",
     "KKN:p25|KKN:p26", "list", "hard", "yes"),

    ("Sanksi apa saja yang bisa dijatuhkan kepada mahasiswa kukerta?",
     "Peringatan secara lisan, peringatan secara tertulis, pengurangan nilai kukerta, perpanjangan masa kukerta, serta penarikan dari lokasi sebelum masa berakhirnya kukerta dan dinyatakan gugur.",
     "KKN:p26", "list", "medium", "yes"),

    ("Apa saja tugas dosen pembimbing lapangan kukerta?",
     "Membimbing minimal 3 dan maksimal 5 kelompok, berkoordinasi dengan koordinator lokasi, menjalin komunikasi pada kegiatan observasi pendahuluan, melakukan pembimbingan atau monitoring minimal 4 kali luring, menyerahkan dan menjemput mahasiswa, membantu menyelesaikan permasalahan mahasiswa, menjalin kerja sama dengan tokoh dan lembaga di lokasi, berkonsultasi dengan koordinator lokasi bila masalahnya kompleks, mengoreksi dan mengesahkan laporan, memberikan penilaian luaran 5 hari sebelum penjemputan, mengunggah nilai 15 hari setelah penarikan, dan berpartisipasi aktif dalam rapat LPPM.",
     "KKN:p13", "list", "hard", "yes"),

    ("Apa saja syarat menjadi dosen pembimbing lapangan kukerta?",
     "Memiliki NIDN atau NUPTK, berstatus dosen aktif Universitas Riau atau tidak sedang tugas dan izin belajar, data NIP, NIDN, atau NUPTK terdaftar di Portal KUKERTA Unri, bersedia ditempatkan pada lokasi desa yang ditetapkan LPPM, bersedia mengikuti seluruh ketentuan pelaksanaan kukerta, mengajukan usulan proposal pengabdian kepada masyarakat saat mendaftar, dan untuk Kukerta Berdampak Khusus berasal dari dosen yang diusulkan Fakultas.",
     "KKN:p13", "list", "hard", "yes"),

    ("Skema penelitian apa saja yang disediakan LPPM UNRI?",
     "Riset Afirmasi, Riset Peningkatan Kapasitas Dosen Muda, Riset Kolaborasi Internasional, Riset Unggulan Universitas Riau, Riset Mandatory, Riset Produk dan Prototipe, serta Riset Inovasi dan Hilirisasi Industri.",
     "PKM:p4|PKM:p15", "list", "medium", "yes"),

    ("Skema pengabdian apa saja yang disediakan LPPM UNRI?",
     "Pemberdayaan Berbasis Masyarakat, Pemberdayaan Desa Binaan, dan Pemberdayaan Mandatory.",
     "PKM:p4|PKM:p52", "list", "easy", "yes"),

    ("Unit penunjang akademik apa saja yang ada di Universitas Riau?",
     "Perpustakaan, Teknologi Informasi dan Komunikasi, Bahasa, Laboratorium Terpadu, Bimbingan dan Konseling, Pengembangan Karier dan Kewirausahaan, Layanan Uji Kompetensi, serta Percetakan dan Penerbitan.",
     "PDS16:p14", "list", "medium", "yes"),

    ("Tema apa saja yang bisa dipilih pada program FISIP Berdampak untuk Negeri?",
     "Antara lain Desa Literasi, Desa Pintar, Desa Sejarah, Desa Sejahtera, Desa Sadar Hukum, Desa Digital, Desa TOGA, Enterpreneur dan Industri Kreatif, Mitigasi Bencana, Eco Eduwisata, Lingkungan Berkelanjutan, Pengolahan Sampah, Pemberdayaan ekonomi masyarakat, Penguatan tata kelola desa atau komunitas, Edukasi sosial dan literasi, Transformasi digital masyarakat, Kewirausahaan dan Ekonomi Kreatif, Stunting dan Gizi Seimbang, serta Literasi Kesehatan.",
     "FISIP:p7", "list", "hard", "yes"),

    ("Apa saja yang dimuat di dalam ijazah?",
     "Nomor ijazah nasional, lambang perguruan tinggi, nama perguruan tinggi, nomor keputusan akreditasi perguruan tinggi dan/atau program studi, Program Pendidikan Tinggi, nama program studi, nama lengkap pemilik ijazah, tempat dan tanggal lahir, nomor pokok mahasiswa, nomor induk kependudukan atau nomor paspor, gelar beserta singkatannya, tanggal kelulusan, tempat dan tanggal penerbitan, nama dan jabatan pimpinan perguruan tinggi yang menandatangani, stempel perguruan tinggi, dan foto pemilik ijazah.",
     "IJZ22:p4|IJZ22:p5", "list", "hard", "yes"),

    ("Apa saja tata tertib mahasiswa selama mengikuti PKKMB di UNRI?",
     "Mengikuti kegiatan pukul 07.00 sampai 16.30 WIB kecuali hari ketiga sampai 17.00 WIB, mengisi presensi setiap datang dan pulang, dilarang melakukan tindakan provokatif, wajib berpakaian rapi baju putih dan celana atau rok hitam, dasi, dan jas almamater, wajib menggunakan identitas pengenal diri, dilarang berkata atau menulis kata tidak sopan, wajib membuat resume setiap materi sepanjang satu halaman sekitar 250 kata, mengikuti arahan relawan resmi, mengikuti kegiatan dari awal sampai akhir, disarankan membawa air minum dan perlengkapan pribadi, serta memarkir kendaraan di tempat yang disediakan.",
     "PKKMB:p10", "list", "hard", "yes"),

    ("Apa saja tahapan tutorial pembelajaran berbasis masalah di FK UNRI?",
     "Identifikasi dan klarifikasi istilah yang belum dimengerti dalam skenario, menentukan masalah, analisis masalah melalui curah pendapat, membuat kajian sistematis penjelasan hasil langkah sebelumnya, menyusun tujuan pembelajaran, belajar mandiri mengumpulkan informasi tambahan, dan saling bertukar informasi yang didapat selama belajar mandiri.",
     "FKU:p31|FKU:p32", "list", "hard", "yes"),

    ("Materi apa saja yang diberikan pada kegiatan PKKMB di UNRI?",
     "Kehidupan berbangsa, bernegara, dan pembinaan kesadaran bela negara, pengenalan sistem pendidikan tinggi di Indonesia, perguruan tinggi di era revolusi industri 4.0 dan society 5.0, pengenalan growth mindset mahasiswa, pengembangan karakter mahasiswa, pengenalan keselamatan, kesehatan kerja, dan lingkungan, serta materi lain yang dipandang perlu sesuai kebutuhan mahasiswa dan perguruan tinggi.",
     "PKKMB:p3", "list", "medium", "yes"),

    ("Prinsip apa saja yang harus dipenuhi pelaksanaan kukerta?",
     "Empat prinsip, yaitu dapat dilaksanakan atau feasible, dapat diterima atau acceptable, partisipatif atau participative, dan berkesinambungan atau sustainable.",
     "KKN:p4", "list", "easy", "yes"),

    # ---- definisi ----
    ("Kukerta Berdampak itu apa?",
     "Kegiatan pengabdian mahasiswa kepada masyarakat yang menghasilkan perubahan nyata dan berkelanjutan di lingkungan tempat dilaksanakannya kegiatan kukerta, dan merupakan bagian dari program Kampus Berdampak yang dicanangkan Kementerian Pendidikan Tinggi, Sains dan Teknologi.",
     "KKN:p5", "definisi", "medium", "yes"),

    ("Apa yang dimaksud dengan gratifikasi?",
     "Pemberian dalam arti luas, yakni uang, barang, rabat atau discount, komisi, pinjaman tanpa bunga, tiket perjalanan, fasilitas penginapan, perjalanan wisata, pengobatan cuma-cuma, dan fasilitas lainnya, baik yang diterima di dalam negeri maupun di luar negeri, yang dilakukan dengan menggunakan sarana elektronik atau tanpa sarana elektronik.",
     "GRAT:p3", "definisi", "medium", "yes"),

    ("Pangkalan Data Pendidikan Tinggi itu apa?",
     "Sistem yang menghimpun data pendidikan tinggi dari seluruh perguruan tinggi yang terintegrasi secara nasional.",
     "PDD:p1", "definisi", "easy", "yes"),

    ("Surat Keterangan Pendamping Ijazah itu apa?",
     "Dokumen yang diterbitkan oleh perguruan tinggi yang memuat informasi tentang pemenuhan kompetensi lulusan pendidikan akademik dan vokasi.",
     "IJZ22:p2", "definisi", "easy", "yes"),

    ("Apa yang dimaksud Penomoran Ijazah Nasional?",
     "Sistem penomoran ijazah yang diberlakukan secara nasional dengan menggunakan format penomoran yang diterbitkan oleh Kementerian.",
     "IJZ22:p3", "definisi", "easy", "yes"),

    ("Jalur SMBT di UNRI itu apa?",
     "Seleksi Masuk Berbasis Tes, yaitu seleksi yang dilaksanakan UNRI untuk menjaring potensi lulusan SMA, SMK, dan MA yang unggul khususnya untuk wilayah Riau, dengan metode Ujian Tulis Berbasis Komputer dan kriteria lain yang ditetapkan berdasarkan Keputusan Rektor.",
     "SMBT:p1", "definisi", "easy", "yes"),

    ("Apa arti pengabdian kepada masyarakat dalam aturan Universitas Riau?",
     "Kegiatan civitas akademika untuk pengamalan ilmu pengetahuan, teknologi dan seni budaya langsung pada masyarakat secara kelembagaan melalui metodologi ilmiah sebagai penyebaran Tri Dharma Perguruan Tinggi serta tanggung jawab yang luhur dalam usaha mengembangkan kemampuan masyarakat.",
     "PGB:p1", "definisi", "medium", "yes"),

    ("Tutorial dalam pembelajaran di FK UNRI itu apa?",
     "Kegiatan akademik terstruktur yang dilaksanakan mahasiswa dalam bentuk diskusi kelompok untuk mencapai tujuan pembelajaran sesuai Standar Kompetensi Dokter, dengan mahasiswa dibagi dalam grup berjumlah lebih kurang 7 sampai 12 orang yang difasilitasi oleh tutor.",
     "FKU:p31", "definisi", "medium", "yes"),

    ("Portofolio dalam pendaftaran jalur mandiri UNRI itu apa?",
     "Dokumentasi kumpulan karya atau penampilan siswa yang dibuat selama mengikuti pendidikan menengah.",
     "SMBT:p13", "definisi", "easy", "yes"),

    # ---- factoid ----
    ("Universitas Riau masuk klaster apa dalam kinerja penelitian dan pengabdian?",
     "Klaster Mandiri tahun 2025.",
     "PKM:p3", "factoid", "easy", "yes"),

    ("Berapa persen dana penerimaan yang dialokasikan untuk penelitian dan pengabdian?",
     "15 persen dari total dana penerimaan PNBP.",
     "PKM:p4", "factoid", "easy", "yes"),

    ("Satu dosen pembimbing lapangan kukerta boleh membimbing berapa mahasiswa?",
     "Maksimal 40 orang mahasiswa di lokasi kukerta, dengan membimbing minimal 3 dan maksimal 5 kelompok.",
     "KKN:p13", "factoid", "easy", "yes"),

    ("Berapa jam sehari yang disarankan untuk kegiatan kukerta?",
     "Berkisar 5 sampai 6 jam tiap hari.",
     "KKN:p8", "factoid", "easy", "yes"),

    ("Berapa persen kehadiran pembekalan kukerta supaya dinyatakan lulus?",
     "Minimal 90 persen, ditambah mengikuti pembekalan dengan tertib dan disiplin serta lulus pendalaman materi berupa pretest dan posttest.",
     "KKN:p11", "factoid", "medium", "yes"),

    ("Berapa biaya pendaftaran jalur mandiri Universitas Riau?",
     "Rp350.000.",
     "SMBT:p12", "factoid", "easy", "yes"),

    ("Kapan UTBK jalur mandiri UNRI 2026 dilaksanakan?",
     "Tanggal 9 sampai 14 Juli 2026.",
     "SMBT:p12|SMBT:p13", "factoid", "easy", "yes"),

    ("Berapa lama masa studi maksimal mahasiswa kedokteran UNRI?",
     "Maksimal 12 semester.",
     "FKU:p24", "factoid", "easy", "yes"),

    ("Berapa batas nilai pemberian saat pesta pernikahan yang tidak wajib dilaporkan sebagai gratifikasi?",
     "Paling banyak Rp1.000.000 per pemberi dalam setiap acara.",
     "GRAT:p4", "factoid", "medium", "yes"),

    ("Gaji pokok dosen dengan jabatan Lektor merujuk pada golongan berapa?",
     "Golongan III/c.",
     "PMD44:p27", "factoid", "medium", "yes"),

    ("Berapa dana maksimal skema Pemberdayaan Mandatory dan berapa anggota dosen minimalnya?",
     "Usulan dana ke LPPM maksimal Rp40.000.000 dengan anggota pengusul minimal 2 orang dosen.",
     "PKM:p28", "factoid", "medium", "yes"),

    ("Berapa anggaran yang boleh diinvestasikan ke mitra pada skema desa binaan?",
     "Maksimal 20 persen dari total anggaran yang diajukan, dalam bentuk belanja barang.",
     "PKM:p26", "factoid", "medium", "yes"),
]


def semua_baris() -> list[tuple]:
    """Gabungkan seluruh kategori menjadi satu daftar baris."""
    baris = list(KOMPARATIF) + list(MULTI_HOP) + list(PROSEDUR) + list(LAINNYA)
    baris += [(q, "Informasi tersebut tidak tersedia dalam dokumen.", "",
               "unanswerable", "hard", "no") for q in UNANSWERABLE]
    return baris


NEW_ROWS_BAL = semua_baris()
