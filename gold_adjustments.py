"""gold_adjustments.py - penyesuaian gold_label setelah korpus diperluas.

GOLD_ANY  : pertanyaan yang entri gold-nya adalah LOKASI ALTERNATIF, cukup
            ditemukan salah satu (rank_metrics gold_mode='any').
GOLD_EXTRA: entri gold tambahan, karena jawaban yang sama ternyata juga
            ada di dokumen lain di korpus.

Dibangkitkan dari check_gold_conflicts.py; kunci = teks pertanyaan.
"""

GOLD_ANY = {
    'Apa yang dimaksud dengan dosen?',
    'Apa tugas Dosen Penasihat Akademik?',
    'Apa itu transfer kredit?',
    'Apa bedanya double degree dan joint degree?',
    'Satu semester itu berapa minggu?',
    'Satu SKS itu setara berapa jam kegiatan?',
    'Apakah UNRI punya aturan tentang penyetaraan kredit akademik internasional?',
    'Kapan aturan penyetaraan kredit akademik internasional ditetapkan?',
    'Apa itu ECTS?',
    'Satu kredit ECTS setara berapa SKS?',
    'Program Sarjana 144 SKS setara berapa ECTS?',
    'Berapa beban ECTS untuk Program Magister?',
    'Berapa beban ECTS untuk Program Doktor?',
    'Satu SKS setara berapa jam dalam perhitungan konversi ECTS?',
    'Berapa jam belajar yang ditetapkan sebagai standar minimum untuk 1 ECTS?',
    'Nilai ECTS A setara dengan nilai apa di UNRI?',
    'Nilai ECTS FX setara dengan nilai apa di UNRI?',
    'Bagaimana kalau perguruan tinggi luar negeri tidak memakai sistem ECTS?',
    'Apa yang dimaksud dengan Capaian Pembelajaran Lulusan?',
    'Berapa kali minimal saya harus konsultasi dengan Penasihat Akademik dalam satu semester?',
    'Universitas Riau itu perguruan tinggi yang seperti apa?',
    'Apa saja yang dinilai saat ujian tugas akhir?',
    'Jatah SKS semester depan ditentukan dari IPS semester mana?',
    'Berapa maksimal mahasiswa bimbingan bagi Komisi Pembimbing Program Magister?',
    'Berapa jumlah penguji pada ujian tesis Program Magister?',
    'Apa dasar hukum organisasi dan tata kerja Universitas Riau?',
    'Apa saja syarat untuk mendaftar sebagai mahasiswa baru di UNRI?',
    'Jenjang apa saja yang termasuk pendidikan tinggi?',
    'Berapa lama dosen harus menginput nilai setelah ujian selesai?',
    'Berapa orang pembimbing tugas akhir untuk program doktor dan D3?',
    'Berapa lama masa bimbingan tugas akhir?',
}

GOLD_EXTRA = {
    # Q188
    'Berapa kali minimal saya harus konsultasi dengan Penasihat Akademik dalam satu semester?':
        'pertor_3_2015_Peraturan-Akademik.pdf:p26',
    # Q003
    'Universitas Riau itu perguruan tinggi yang seperti apa?':
        'pertor_3_2015_Peraturan-Akademik.pdf',
    # Q191
    'Apa saja yang dinilai saat ujian tugas akhir?':
        'pertor_3_2015_Peraturan-Akademik.pdf:p20',
    # Q167
    'Jatah SKS semester depan ditentukan dari IPS semester mana?':
        'pertor_3_2015_Peraturan-Akademik.pdf:p8',
    # Q318
    'Berapa maksimal mahasiswa bimbingan bagi Komisi Pembimbing Program Magister?':
        'pertor_3_2015_Peraturan-Akademik.pdf:p28',
    # Q324
    'Berapa jumlah penguji pada ujian tesis Program Magister?':
        'pertor_3_2015_Peraturan-Akademik.pdf:p20',
    # Q022
    'Apa dasar hukum organisasi dan tata kerja Universitas Riau?':
        'pertor_3_2015_Peraturan-Akademik.pdf',
    # Q006
    'Apa saja syarat untuk mendaftar sebagai mahasiswa baru di UNRI?':
        'pertor_3_2015_Peraturan-Akademik.pdf',
    # Q002
    'Jenjang apa saja yang termasuk pendidikan tinggi?':
        'pertor_3_2015_Peraturan-Akademik.pdf',
    # Q183
    'Berapa lama dosen harus menginput nilai setelah ujian selesai?':
        'pertor_3_2015_Peraturan-Akademik.pdf:p32',
    # Q181
    'Berapa orang pembimbing tugas akhir untuk program doktor dan D3?':
        'pertor_3_2015_Peraturan-Akademik.pdf:p27',
    # Q180
    'Berapa lama masa bimbingan tugas akhir?':
        'pertor_3_2015_Peraturan-Akademik.pdf:p27',
}
