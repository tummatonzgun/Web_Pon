import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import os

class DIE_ATTACH_UPH_Analyzer:
    def __init__(self):
        self.raw_data = None
        self.clean_data = None
        self.stats_report = None
        
    def load_data(self, file_path):
        """โหลดข้อมูล Die Attach จากไฟล์"""
        try:
            if file_path.endswith('.csv'):
                self.raw_data = pd.read_csv(file_path, encoding='utf-8-sig')
            else:
                self.raw_data = pd.read_excel(file_path)
            
            # ทำความสะอาดชื่อคอลัมน์
            self.raw_data.columns = self.raw_data.columns.str.strip().str.lower()
            print("โหลดข้อมูล Die Attach สำเร็จ!")
            return True
        except Exception as e:
            print(f"เกิดข้อผิดพลาดในการโหลดข้อมูล: {e}")
            return False
    
    def validate_data(self):
        """ตรวจสอบความถูกต้องของข้อมูล"""
        required_columns = ['uph', 'machine', 'device', 'lot_no', 'timestamp']
        missing_cols = [col for col in required_columns if col not in self.raw_data.columns]
        
        if missing_cols:
            print(f"ขาดคอลัมน์ที่จำเป็น: {missing_cols}")
            return False
            
        # แปลงประเภทข้อมูล
        self.raw_data['uph'] = pd.to_numeric(self.raw_data['uph'], errors='coerce')
        self.raw_data['timestamp'] = pd.to_datetime(self.raw_data['timestamp'])
        
        # ลบแถวที่มีค่าว่าง
        initial_count = len(self.raw_data)
        self.raw_data = self.raw_data.dropna(subset=['uph', 'machine', 'device'])
        cleaned_count = len(self.raw_data)
        
        print(f"ทำความสะอาดข้อมูล: ลบ {initial_count - cleaned_count} แถวที่มีค่าว่าง")
        return True
    
    def detect_outliers(self, method='iqr'):
        """ตรวจจับและลบ Outliers"""
        self.clean_data = self.raw_data.copy()
        
        # คำนวณ Outliers แยกตามเครื่องและอุปกรณ์
        grouped = self.clean_data.groupby(['machine', 'device'])
        
        for (machine, device), group in grouped:
            uph_values = group['uph']
            
            if method == 'iqr':
                # ใช้ IQR method
                Q1 = uph_values.quantile(0.25)
                Q3 = uph_values.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
            elif method == 'zscore':
                # ใช้ Z-Score method
                z_scores = np.abs(stats.zscore(uph_values))
                lower_bound = uph_values.mean() - 3 * uph_values.std()
                upper_bound = uph_values.mean() + 3 * uph_values.std()
            
            # ทำเครื่องหมาย Outliers
            condition = (uph_values < lower_bound) | (uph_values > upper_bound)
            outlier_indices = group[condition].index
            
            # อัปเดตข้อมูล
            self.clean_data.loc[outlier_indices, 'is_outlier'] = True
            self.clean_data['is_outlier'] = self.clean_data['is_outlier'].fillna(False)
        
        print(f"พบและทำเครื่องหมาย Outliers ทั้งหมด {self.clean_data['is_outlier'].sum()} แถว")
        return True
    
    def remove_outliers(self):
        """ลบแถวที่ถูกระบุว่าเป็น Outliers"""
        if 'is_outlier' not in self.clean_data.columns:
            print("ยังไม่ได้ทำการตรวจจับ Outliers")
            return False
            
        before_count = len(self.clean_data)
        self.clean_data = self.clean_data[~self.clean_data['is_outlier']]
        after_count = len(self.clean_data)
        
        print(f"ลบ Outliers สำเร็จ: ลบ {before_count - after_count} แถว")
        return True
    
    def calculate_da_uph_stats(self):
        """คำนวณสถิติ UPH สำหรับ Die Attach"""
        if self.clean_data is None:
            print("ไม่มีข้อมูลที่ทำความสะอาดแล้ว")
            return False
            
        # คำนวณสถิติพื้นฐาน
        stats_df = self.clean_data.groupby(['machine', 'device'])['uph'].agg([
            ('count', 'count'),
            ('mean', 'mean'),
            ('median', 'median'),
            ('std_dev', 'std'),
            ('min', 'min'),
            ('max', 'max'),
            ('25%', lambda x: x.quantile(0.25)),
            ('75%', lambda x: x.quantile(0.75))
        ]).reset_index()
        
        # คำนวณ Cp, Cpk
        for index, row in stats_df.iterrows():
            usl = row['mean'] + 3 * row['std_dev']
            lsl = row['mean'] - 3 * row['std_dev']
            stats_df.at[index, 'Cp'] = (usl - lsl) / (6 * row['std_dev'])
            stats_df.at[index, 'Cpk'] = min(
                (row['mean'] - lsl) / (3 * row['std_dev']),
                (usl - row['mean']) / (3 * row['std_dev'])
            )
        
        self.stats_report = stats_df.round(2)
        return True
    
    def generate_da_report(self, output_folder='DA_UPH_Reports'):
        """สร้างรายงานและกราฟ"""
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        # บันทึกไฟล์ Excel
        report_path = os.path.join(output_folder, 'DA_UPH_Report.xlsx')
        with pd.ExcelWriter(report_path) as writer:
            self.stats_report.to_excel(writer, sheet_name='UPH_Stats', index=False)
            self.clean_data.to_excel(writer, sheet_name='Cleaned_Data', index=False)
        
        # สร้างกราฟ
        self.plot_da_uph_trend(os.path.join(output_folder, 'DA_UPH_Trend.png'))
        self.plot_da_uph_distribution(os.path.join(output_folder, 'DA_UPH_Distribution.png'))
        
        print(f"สร้างรายงานสำเร็จที่โฟลเดอร์: {output_folder}")
        return True
    
    def plot_da_uph_trend(self, save_path=None):
        """สร้างกราฟแสดงแนวโน้ม UPH"""
        plt.figure(figsize=(12, 6))
        
        for machine in self.clean_data['machine'].unique():
            machine_data = self.clean_data[self.clean_data['machine'] == machine]
            for device in machine_data['device'].unique():
                device_data = machine_data[machine_data['device'] == device]
                plt.plot(device_data['timestamp'], device_data['uph'], 
                        'o-', label=f'{machine} - {device}')
        
        plt.title('Die Attach UPH Trend')
        plt.xlabel('Date/Time')
        plt.ylabel('UPH')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_da_uph_distribution(self, save_path=None):
        """สร้างกราฟการกระจายตัวของ UPH"""
        plt.figure(figsize=(12, 6))
        
        for machine in self.clean_data['machine'].unique():
            machine_data = self.clean_data[self.clean_data['machine'] == machine]
            for device in machine_data['device'].unique():
                device_data = machine_data[machine_data['device'] == device]
                plt.hist(device_data['uph'], alpha=0.5, 
                        label=f'{machine} - {device}', bins=20)
        
        plt.title('Die Attach UPH Distribution')
        plt.xlabel('UPH')
        plt.ylabel('Frequency')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

# ตัวอย่างการใช้งาน
if __name__ == "__main__":
    analyzer = DIE_ATTACH_UPH_Analyzer()
    
    # โหลดข้อมูล (เปลี่ยน path เป็นไฟล์ของคุณ)
    data_file = "die_attach_data.xlsx"  # หรือ .csv
    if analyzer.load_data(data_file) and analyzer.validate_data():
        # ตรวจจับและลบ Outliers
        analyzer.detect_outliers(method='iqr')
        analyzer.remove_outliers()
        
        # คำนวณสถิติ
        analyzer.calculate_da_uph_stats()
        
        # สร้างรายงาน
        analyzer.generate_da_report()
        
        # แสดงตัวอย่างผลลัพธ์
        print("\nตัวอย่างสถิติ UPH:")
        print(analyzer.stats_report.head())