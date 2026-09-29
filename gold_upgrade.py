"""gold_upgrade.py - menaikkan gold level dokumen menjadi level halaman.

40 pertanyaan lama semula bergold level dokumen, sehingga tidak bisa ikut
evaluasi level halaman dan membuat dokumennya terbaca 'nol halaman diuji'
pada analisis cakupan. Halaman ditentukan dengan mencocokkan frasa
berurutan terpanjang dari jawaban ke tiap halaman dokumen gold; sepuluh
kasus yang tidak konklusif ditetapkan manual setelah membaca pasalnya.

ANSWER_FIX memuat jawaban acuan yang ternyata KELIRU pada dataset asli.
Ketiganya diverifikasi langsung ke pasal yang bersangkutan.
"""

GOLD_PAGE = {
    # Q001  Untuk apa pedoman akademik Universitas Riau dibuat?
    'Untuk apa pedoman akademik Universitas Riau dibuat?':
        'pertor_4_2021.pdf:p0',
    # Q002  Jenjang apa saja yang termasuk pendidikan tinggi?
    'Jenjang apa saja yang termasuk pendidikan tinggi?':
        'pertor_4_2021.pdf:p4|pertor_3_2015_Peraturan-Akademik.pdf:p1',
    # Q003  Universitas Riau itu perguruan tinggi yang seperti apa?
    'Universitas Riau itu perguruan tinggi yang seperti apa?':
        'pertor_4_2021.pdf:p4|pertor_3_2015_Peraturan-Akademik.pdf:p2',
    # Q004  Status mahasiswa di UNRI ada apa saja?
    'Status mahasiswa di UNRI ada apa saja?':
        'pertor_4_2021.pdf:p11',
    # Q005  Paling lama berapa tahun saya boleh kuliah S1?
    'Paling lama berapa tahun saya boleh kuliah S1?':
        'pertor_4_2021.pdf:p13',
    # Q006  Apa saja syarat untuk mendaftar sebagai mahasiswa baru di UNRI
    'Apa saja syarat untuk mendaftar sebagai mahasiswa baru di UNRI?':
        'pertor_4_2021.pdf:p14|pertor_3_2015_Peraturan-Akademik.pdf:p6',
    # Q007  Maksimal berapa SKS yang boleh saya ambil dalam satu semester 
    'Maksimal berapa SKS yang boleh saya ambil dalam satu semester di S1?':
        'pertor_4_2021.pdf:p17',
    # Q008  Bagaimana cara menghitung IPK?
    'Bagaimana cara menghitung IPK?':
        'pertor_4_2021.pdf:p17',
    # Q009  Mata kuliah wajib di UNRI apa saja?
    'Mata kuliah wajib di UNRI apa saja?':
        'pertor_4_2021.pdf:p24',
    # Q010  KUKERTA itu apa?
    'KUKERTA itu apa?':
        'pertor_4_2021.pdf:p23',
    # Q011  Untuk apa aturan tentang profesi, karier, dan penghasilan dose
    'Untuk apa aturan tentang profesi, karier, dan penghasilan dosen dibuat?':
        'PERMENDIKBUD-RISTEK-44-2024.pdf:p0',
    # Q012  Apa yang dimaksud dengan Tridharma Perguruan Tinggi?
    'Apa yang dimaksud dengan Tridharma Perguruan Tinggi?':
        'PERMENDIKBUD-RISTEK-44-2024.pdf:p1',
    # Q013  Status dosen itu ada apa saja?
    'Status dosen itu ada apa saja?':
        'PERMENDIKBUD-RISTEK-44-2024.pdf:p2',
    # Q014  Jenjang jabatan akademik untuk dosen tetap apa saja?
    'Jenjang jabatan akademik untuk dosen tetap apa saja?':
        'PERMENDIKBUD-RISTEK-44-2024.pdf:p3',
    # Q015  Minimal lulusan apa untuk bisa jadi dosen di program sarjana?
    'Minimal lulusan apa untuk bisa jadi dosen di program sarjana?':
        'PERMENDIKBUD-RISTEK-44-2024.pdf:p4',
    # Q016  Kompetensi apa saja yang harus dimiliki seorang dosen?
    'Kompetensi apa saja yang harus dimiliki seorang dosen?':
        'PERMENDIKBUD-RISTEK-44-2024.pdf:p4',
    # Q017  Apa syaratnya supaya dosen bisa ikut sertifikasi pendidik?
    'Apa syaratnya supaya dosen bisa ikut sertifikasi pendidik?':
        'PERMENDIKBUD-RISTEK-44-2024.pdf:p7',
    # Q018  Uji kompetensi untuk sertifikasi dosen bentuknya seperti apa?
    'Uji kompetensi untuk sertifikasi dosen bentuknya seperti apa?':
        'PERMENDIKBUD-RISTEK-44-2024.pdf:p7',
    # Q019  Bagaimana karier dosen dibina dan dikembangkan di perguruan ti
    'Bagaimana karier dosen dibina dan dikembangkan di perguruan tinggi?':
        'PERMENDIKBUD-RISTEK-44-2024.pdf:p10',
    # Q020  Apa syarat untuk naik jadi Profesor?
    'Apa syarat untuk naik jadi Profesor?':
        'PERMENDIKBUD-RISTEK-44-2024.pdf:p12',
    # Q021  Untuk apa aturan tentang organisasi dan tata kerja Universitas
    'Untuk apa aturan tentang organisasi dan tata kerja Universitas Riau dibuat?':
        '2025permendiktisaintek016.pdf:p0',
    # Q022  Apa dasar hukum organisasi dan tata kerja Universitas Riau?
    'Apa dasar hukum organisasi dan tata kerja Universitas Riau?':
        '2025permendiktisaintek016.pdf:p0|pertor_3_2015_Peraturan-Akademik.pdf:p0',
    # Q023  Universitas Riau itu perguruan tinggi di bawah kementerian apa
    # p25 hanya lampiran tanda tangan; definisinya ada di Pasal 1 angka 2.
    'Universitas Riau itu perguruan tinggi di bawah kementerian apa?':
        '2025permendiktisaintek016.pdf:p1',
    # Q024  Apa saja fungsi Universitas Riau?
    # p4-p5 memuat daftar fakultas, bukan fungsi; fungsi ada di Pasal 4.
    'Apa saja fungsi Universitas Riau?':
        '2025permendiktisaintek016.pdf:p2',
    # Q025  Organisasi Universitas Riau terdiri dari unsur apa saja?
    'Organisasi Universitas Riau terdiri dari unsur apa saja?':
        '2025permendiktisaintek016.pdf:p2',
    # Q026  Siapa yang memimpin Universitas Riau?
    'Siapa yang memimpin Universitas Riau?':
        '2025permendiktisaintek016.pdf:p3',
    # Q027  Wakil Rektor membidangi apa saja?
    # p15 hanya menyebut satu bidang sambil lalu; daftarnya di Pasal 9.
    'Wakil Rektor membidangi apa saja?':
        '2025permendiktisaintek016.pdf:p3',
    # Q028  Apa tugas fakultas di Universitas Riau?
    # p6 memuat tugas jurusan, bukan fakultas; tugas fakultas di Pasal 12.
    'Apa tugas fakultas di Universitas Riau?':
        '2025permendiktisaintek016.pdf:p4',
    # Q029  Universitas Riau punya fakultas apa saja?
    'Universitas Riau punya fakultas apa saja?':
        '2025permendiktisaintek016.pdf:p4|2025permendiktisaintek016.pdf:p5',
    # Q030  Jurusan di fakultas tugasnya apa?
    'Jurusan di fakultas tugasnya apa?':
        '2025permendiktisaintek016.pdf:p6',
    # Q031  Untuk apa aturan tata naskah dinas dibuat?
    'Untuk apa aturan tata naskah dinas dibuat?':
        'TATA NASKAH DINAS_Permendikbud_Nomor_3_Tahun_2021_tentang_TND_Kemdikbud_Full.pdf:p0',
    # Q032  Apa yang dimaksud dengan naskah dinas?
    'Apa yang dimaksud dengan naskah dinas?':
        'TATA NASKAH DINAS_Permendikbud_Nomor_3_Tahun_2021_tentang_TND_Kemdikbud_Full.pdf:p2',
    # Q033  Tata naskah dinas itu apa?
    'Tata naskah dinas itu apa?':
        'TATA NASKAH DINAS_Permendikbud_Nomor_3_Tahun_2021_tentang_TND_Kemdikbud_Full.pdf:p2',
    # Q034  Naskah dinas dibagi menjadi jenis apa saja?
    'Naskah dinas dibagi menjadi jenis apa saja?':
        'TATA NASKAH DINAS_Permendikbud_Nomor_3_Tahun_2021_tentang_TND_Kemdikbud_Full.pdf:p4',
    # Q035  Yang termasuk naskah dinas arahan apa saja?
    'Yang termasuk naskah dinas arahan apa saja?':
        'TATA NASKAH DINAS_Permendikbud_Nomor_3_Tahun_2021_tentang_TND_Kemdikbud_Full.pdf:p2',
    # Q036  Yang termasuk naskah dinas korespondensi apa saja?
    'Yang termasuk naskah dinas korespondensi apa saja?':
        'TATA NASKAH DINAS_Permendikbud_Nomor_3_Tahun_2021_tentang_TND_Kemdikbud_Full.pdf:p9',
    # Q037  Yang termasuk naskah dinas khusus apa saja?
    'Yang termasuk naskah dinas khusus apa saja?':
        'TATA NASKAH DINAS_Permendikbud_Nomor_3_Tahun_2021_tentang_TND_Kemdikbud_Full.pdf:p10',
    # Q038  Siapa yang berwenang menandatangani naskah dinas?
    'Siapa yang berwenang menandatangani naskah dinas?':
        'TATA NASKAH DINAS_Permendikbud_Nomor_3_Tahun_2021_tentang_TND_Kemdikbud_Full.pdf:p28',
    # Q039  Bagaimana cara menomori naskah dinas?
    'Bagaimana cara menomori naskah dinas?':
        'TATA NASKAH DINAS_Permendikbud_Nomor_3_Tahun_2021_tentang_TND_Kemdikbud_Full.pdf:p21',
    # Q040  Bahasa seperti apa yang harus dipakai dalam naskah dinas?
    'Bahasa seperti apa yang harus dipakai dalam naskah dinas?':
        'TATA NASKAH DINAS_Permendikbud_Nomor_3_Tahun_2021_tentang_TND_Kemdikbud_Full.pdf:p17',
    # Q242  Masa studi maksimal S1 di pedoman akademik lama dan di aturan 
    'Masa studi maksimal S1 di pedoman akademik lama dan di aturan terbaru sama atau berbeda?':
        'pertor_4_2021.pdf:p13|9-Perubahan-atas-Peraturan-Rektor-Nomor-8-Tahun-2024-tentang-Penyelenggaraan-Pendidikan-Universitas-Riau.pdf:p11',
}

ANSWER_FIX = {
    # Q036  Pasal 11: korespondensi = nota dinas, surat dinas, surat undangan
    'Yang termasuk naskah dinas korespondensi apa saja?':
        'Nota dinas, surat dinas, dan surat undangan.',
    # Q037  Pasal 15: khusus = nota kesepahaman s.d. perjanjian internasional
    'Yang termasuk naskah dinas khusus apa saja?':
        'Nota kesepahaman, perjanjian kerja sama dalam negeri, surat kuasa, berita acara, surat keterangan, surat pernyataan, surat pengantar, pengumuman, dan perjanjian internasional.',
    # Q039  Pasal 38: nomor = nomor urut + kode naskah dinas
    'Bagaimana cara menomori naskah dinas?':
        'Nomor naskah dinas terdiri atas nomor urut naskah dinas dan kode naskah dinas.',
    # Q029  Jawaban lama memakai akronim (FISIP, FKIP, FMIPA, FEB) yang sama
    # sekali tidak muncul di dokumen, sehingga tidak bisa dinilai terhadap
    # halaman gold-nya. Diganti dengan nama resmi sesuai Pasal 14.
    'Universitas Riau punya fakultas apa saja?':
        'Fakultas Ilmu Sosial dan Ilmu Politik, Fakultas Keguruan dan Ilmu '
        'Pendidikan, Fakultas Matematika dan Ilmu Pengetahuan Alam, Fakultas '
        'Ekonomi dan Bisnis, Fakultas Pertanian, Fakultas Perikanan dan '
        'Kelautan, Fakultas Teknik, Fakultas Kedokteran, Fakultas Hukum, dan '
        'Fakultas Keperawatan.',
}
