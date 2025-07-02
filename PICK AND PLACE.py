import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

class PNP_UPH_Analyzer:
    def __init__(self):
        self.raw_data = None
        self.clean_data = None
        self.package_data = None
        self.final_report = None
    
    def load_production_data(self, file_path):
        """โหลดข้อมูลการผลิต PNP"""
        try:
            if file_path.endswith('.csv'):
                self.raw_data = pd.read_csv(file_path, encoding='utf-8-sig')
            else:
                self.raw_data = pd.read_excel(file_path)
            
            # ทำความสะอาดชื่อคอลัมน์
            self.raw_data.columns = self.raw_data.columns.str.strip().str.lower()
            
            # ตรวจสอบคอลัมน์ที่จำเป็น
            required_cols = ['package_code', 'machine_model', 'uph', 'timestamp']
            missing_cols = [col for col in required_cols if col not in self.raw_data.columns]
            
            if missing_cols:
                print(f"ขาดคอลัมน์ที่จำเป็น: {missing_cols}")
                return False
            
            # แปลงประเภทข้อมูล
            self.raw_data['uph'] = pd.to_numeric(self.raw_data['uph'], errors='coerce')
            self.raw_data['timestamp'] = pd.to_datetime(self.raw_data['timestamp'])
            self.raw_data['package_code'] = self.raw_data['package_code'].astype(str).str.strip().str.upper()
            
            print("โหลดข้อมูล Pick & Place สำเร็จ!")
            return True
            
        except Exception as e:
            print(f"เกิดข้อผิดพลาดในการโหลดข้อมูล: {e}")
            return False
    
    def load_package_data(self, file_path):
        """โหลดข้อมูล Package เสริม"""
        try:
            self.package_data = pd.read_excel(file_path)
            self.package_data.columns = self.package_data.columns.str.strip().str.upper()
            
            # ตรวจสอบคอลัมน์ที่จำเป็น
            required_cols = ['PACKAGE_CODE', 'LEAD_COUNT', 'BODY_SIZE']
            missing_cols = [col for col in required_cols if col not in self.package_data.columns]
            
            if missing_cols:
                print(f"ขาดคอลัมน์ที่จำเป็นในไฟล์ Package Data: {missing_cols}")
                return False
            
            self.package_data['PACKAGE_CODE'] = self.package_data['PACKAGE_CODE'].astype(str).str.strip().str.upper()
            print("โหลดข้อมูล Package เสริมสำเร็จ!")
            return True
            
        except Exception as e:
            print(f"เกิดข้อผิดพลาดในการโหลดข้อมูล Package: {e}")
            return False
    
    def preprocess_data(self):
        """เตรียมและทำความสะอาดข้อมูล"""
        if self.raw_data is None:
            print("ไม่มีข้อมูลการผลิตให้ประมวลผล")
            return False
        
        # ลบแถวที่มีค่าว่าง
        initial_count = len(self.raw_data)
        self.clean_data = self.raw_data.dropna(subset=['uph', 'machine_model', 'package_code'])
        cleaned_count = len(self.clean_data)
        
        print(f"ทำความสะอาดข้อมูล: ลบ {initial_count - cleaned_count} แถวที่มีค่าว่าง")
        
        # ผสานข้อมูล Package เสริม (ถ้ามี)
        if self.package_data is not None:
            self.clean_data = pd.merge(
                self.clean_data,
                self.package_data,
                left_on='package_code',
                right_on='PACKAGE_CODE',
                how='left'
            )
        
        # คำนวณ UPH ปรับตามขนาด Package (ถ้ามีข้อมูล)
        if 'BODY_SIZE' in self.clean_data.columns:
            self.clean_data['size_adjusted_uph'] = self.clean_data['uph'] / self.clean_data['BODY_SIZE']
        else:
            self.clean_data['size_adjusted_uph'] = self.clean_data['uph']
        
        return True
    
    def detect_and_remove_outliers(self, method='iqr'):
        """ตรวจจับและลบ Outliers"""
        if self.clean_data is None:
            print("ไม่มีข้อมูลที่ทำความสะอาดแล้ว")
            return False
        
        original_count = len(self.clean_data)
        self.clean_data['is_outlier'] = False
        
        # คำนวณ Outliers แยกตาม Package Code และ Machine Model
        grouped = self.clean_data.groupby(['package_code', 'machine_model'])
        
        for (pkg, model), group in grouped:
            uph_values = group['size_adjusted_uph']
            
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
            outlier_condition = (uph_values < lower_bound) | (uph_values > upper_bound)
            outlier_indices = group[outlier_condition].index
            self.clean_data.loc[outlier_indices, 'is_outlier'] = True
        
        # ลบ Outliers
        before_count = len(self.clean_data)
        self.clean_data = self.clean_data[~self.clean_data['is_outlier']]
        after_count = len(self.clean_data)
        
        print(f"ลบ Outliers สำเร็จ: ลบ {before_count - after_count} แถว (จากทั้งหมด {original_count} แถว)")
        return True
    
    def calculate_pnp_stats(self):
        """คำนวณสถิติสำคัญสำหรับ Pick & Place"""
        if self.clean_data is None:
            print("ไม่มีข้อมูลที่ทำความสะอาดแล้ว")
            return False
        
        # สร้าง DataFrame สำหรับรายงาน
        stats_df = self.clean_data.groupby(['package_code', 'machine_model']).agg({
            'size_adjusted_uph': ['mean', 'median', 'std', 'count'],
            'uph': 'mean'
        }).reset_index()
        
        # จัดรูปแบบคอลัมน์ใหม่
        stats_df.columns = [
            'Package Code', 'Machine Model', 
            'Mean Size-Adjusted UPH', 'Median Size-Adjusted UPH', 
            'Std Dev', 'Data Points',
            'Mean Raw UPH'
        ]
        
        # คำนวณ Process Capability (เฉพาะเมื่อมีข้อมูล Body Size)
        if 'BODY_SIZE' in self.clean_data.columns:
            stats_df['Size_Normalized_Std'] = stats_df['Std Dev'] / stats_df['Mean Size-Adjusted UPH']
            stats_df['Cp'] = 1 / (6 * stats_df['Size_Normalized_Std'])
            stats_df['Cpk'] = stats_df['Cp'] * (1 - np.abs(1 - stats_df['Mean Size-Adjusted UPH'] / stats_df['Mean Size-Adjusted UPH'].mean()))
        
        self.final_report = stats_df.round(2)
        return True
    
    def generate_visualizations(self, output_folder='PNP_Analysis_Results'):
        """สร้าง visualization และรายงาน"""
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        # 1. สร้างกราฟ UPH Trend
        self._plot_pnp_trend(output_folder)
        
        # 2. สร้างกราฟเปรียบเทียบ Package
        self._plot_package_comparison(output_folder)
        
        # 3. สร้างกราฟเปรียบเทียบเครื่อง
        self._plot_machine_comparison(output_folder)
        
        # 4. บันทึกไฟล์ Excel
        self._save_excel_report(output_folder)
        
        print(f"\nสร้างรายงานและกราฟสำเร็จที่โฟลเดอร์: {output_folder}")
    
    def _plot_pnp_trend(self, output_folder):
        """สร้างกราฟแสดงแนวโน้ม UPH"""
        plt.figure(figsize=(14, 7))
        
        # แยกสีตาม Package Code
        unique_packages = self.clean_data['package_code'].unique()
        palette = sns.color_palette("husl", len(unique_packages))
        
        for i, pkg in enumerate(unique_packages):
            pkg_data = self.clean_data[self.clean_data['package_code'] == pkg]
            plt.scatter(
                pkg_data['timestamp'], 
                pkg_data['size_adjusted_uph'], 
                color=palette[i],
                label=pkg,
                alpha=0.6
            )
            
            # เส้นแนวโน้ม
            sns.regplot(
                x=pkg_data['timestamp'].astype('int64')//10**9,
                y=pkg_data['size_adjusted_uph'],
                scatter=False,
                color=palette[i],
                line_kws={'linestyle':'--', 'alpha':0.5}
            )
        
        plt.title('PNP Size-Adjusted UPH Trend')
        plt.xlabel('Date/Time')
        plt.ylabel('Size-Adjusted UPH')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()
        
        plot_path = os.path.join(output_folder, 'PNP_UPH_Trend.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_package_comparison(self, output_folder):
        """สร้างกราฟเปรียบเทียบ Package"""
        plt.figure(figsize=(12, 6))
        
        sns.boxplot(
            x=self.clean_data['package_code'],
            y=self.clean_data['size_adjusted_uph'],
            palette='viridis'
        )
        
        plt.title('PNP UPH Comparison by Package Code')
        plt.xlabel('Package Code')
        plt.ylabel('Size-Adjusted UPH')
        plt.xticks(rotation=45)
        plt.grid(True)
        
        plot_path = os.path.join(output_folder, 'PNP_Package_Comparison.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_machine_comparison(self, output_folder):
        """สร้างกราฟเปรียบเทียบประสิทธิภาพเครื่อง"""
        plt.figure(figsize=(12, 6))
        
        machine_stats = self.clean_data.groupby('machine_model')['size_adjusted_uph'].mean().sort_values()
        
        sns.barplot(
            x=machine_stats.index,
            y=machine_stats.values,
            palette='Blues_d'
        )
        
        plt.title('Average PNP UPH by Machine')
        plt.xlabel('Machine Model')
        plt.ylabel('Average Size-Adjusted UPH')
        plt.xticks(rotation=45)
        
        # เพิ่มค่าบนกราฟ
        for i, v in enumerate(machine_stats.values):
            plt.text(i, v + 0.1, f"{v:.2f}", ha='center')
        
        plot_path = os.path.join(output_folder, 'PNP_Machine_Performance.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _save_excel_report(self, output_folder):
        """บันทึกผลลัพธ์เป็นไฟล์ Excel"""
        report_path = os.path.join(output_folder, 'PNP_UPH_Analysis_Report.xlsx')
        
        with pd.ExcelWriter(report_path, engine='openpyxl') as writer:
            # สถิติสรุป
            self.final_report.to_excel(
                writer, 
                sheet_name='Summary Statistics', 
                index=False
            )
            
            # ข้อมูลที่ทำความสะอาดแล้ว
            self.clean_data.to_excel(
                writer, 
                sheet_name='Cleaned Data', 
                index=False
            )
            
            # Outliers ที่ถูกลบ (ถ้ามี)
            if 'is_outlier' in self.raw_data.columns:
                outliers = self.raw_data[self.raw_data['is_outlier']]
                outliers.to_excel(
                    writer, 
                    sheet_name='Removed Outliers', 
                    index=False
                )

# ตัวอย่างการใช้งาน
if __name__ == "__main__":
    analyzer = PNP_UPH_Analyzer()
    
    # โหลดข้อมูลการผลิต PNP (เปลี่ยน path เป็นไฟล์ของคุณ)
    production_file = "pnp_production_data.xlsx"
    if not analyzer.load_production_data(production_file):
        exit()
    
    # โหลดข้อมูล Package เสริม (optional)
    package_file = "package_data.xlsx"
    if os.path.exists(package_file):
        analyzer.load_package_data(package_file)
    
    # ประมวลผลข้อมูล
    if analyzer.preprocess_data():
        # ตรวจจับและลบ Outliers
        analyzer.detect_and_remove_outliers(method='iqr')
        
        # คำนวณสถิติ
        analyzer.calculate_pnp_stats()
        
        # สร้างรายงานและกราฟ
        analyzer.generate_visualizations()
        
        # แสดงตัวอย่างผลลัพธ์
        print("\nตัวอย่างรายงานสรุป:")
        print(analyzer.final_report.head())