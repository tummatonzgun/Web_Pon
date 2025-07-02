import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

class WB_UPH_Analyzer:
    def __init__(self):
        self.raw_data = None
        self.clean_data = None
        self.wire_data = None
        self.final_report = None
    
    def load_production_data(self, file_path):
        """โหลดข้อมูลการผลิต WB"""
        try:
            if file_path.endswith('.csv'):
                self.raw_data = pd.read_csv(file_path, encoding='utf-8-sig')
            else:
                self.raw_data = pd.read_excel(file_path)
            
            # ทำความสะอาดชื่อคอลัมน์
            self.raw_data.columns = self.raw_data.columns.str.strip().str.lower()
            
            # ตรวจสอบคอลัมน์ที่จำเป็น
            required_cols = ['bom_no', 'machine_model', 'uph', 'timestamp', 'wire_count']
            missing_cols = [col for col in required_cols if col not in self.raw_data.columns]
            
            if missing_cols:
                print(f"ขาดคอลัมน์ที่จำเป็น: {missing_cols}")
                return False
            
            # แปลงประเภทข้อมูล
            self.raw_data['uph'] = pd.to_numeric(self.raw_data['uph'], errors='coerce')
            self.raw_data['wire_count'] = pd.to_numeric(self.raw_data['wire_count'], errors='coerce')
            self.raw_data['timestamp'] = pd.to_datetime(self.raw_data['timestamp'])
            self.raw_data['bom_no'] = self.raw_data['bom_no'].astype(str).str.strip().str.upper()
            
            print("โหลดข้อมูล Wire Bond สำเร็จ!")
            return True
            
        except Exception as e:
            print(f"เกิดข้อผิดพลาดในการโหลดข้อมูล: {e}")
            return False
    
    def load_wire_data(self, file_path):
        """โหลดข้อมูล Wire จากไฟล์เสริม"""
        try:
            self.wire_data = pd.read_excel(file_path)
            self.wire_data.columns = self.wire_data.columns.str.strip().str.upper()
            
            # ตรวจสอบคอลัมน์ที่จำเป็น
            required_cols = ['BOM_NO', 'NO_BUMP', 'NUMBER_REQUIRED']
            missing_cols = [col for col in required_cols if col not in self.wire_data.columns]
            
            if missing_cols:
                print(f"ขาดคอลัมน์ที่จำเป็นในไฟล์ Wire Data: {missing_cols}")
                return False
            
            self.wire_data['BOM_NO'] = self.wire_data['BOM_NO'].astype(str).str.strip().str.upper()
            print("โหลดข้อมูล Wire เสริมสำเร็จ!")
            return True
            
        except Exception as e:
            print(f"เกิดข้อผิดพลาดในการโหลดข้อมูล Wire: {e}")
            return False
    
    def preprocess_data(self):
        """เตรียมและทำความสะอาดข้อมูล"""
        if self.raw_data is None:
            print("ไม่มีข้อมูลการผลิตให้ประมวลผล")
            return False
        
        # ลบแถวที่มีค่าว่าง
        initial_count = len(self.raw_data)
        self.clean_data = self.raw_data.dropna(subset=['uph', 'machine_model', 'bom_no', 'wire_count'])
        cleaned_count = len(self.clean_data)
        
        print(f"ทำความสะอาดข้อมูล: ลบ {initial_count - cleaned_count} แถวที่มีค่าว่าง")
        
        # คำนวณ UPH ปรับแล้ว (Adjusted UPH)
        self._calculate_adjusted_uph()
        
        return True
    
    def _calculate_adjusted_uph(self):
        """คำนวณ Adjusted UPH โดยคำนึงถึง Wire Count และ Bump"""
        if self.wire_data is not None:
            # Merge กับข้อมูล Wire
            self.clean_data = pd.merge(
                self.clean_data,
                self.wire_data,
                left_on='bom_no',
                right_on='BOM_NO',
                how='left'
            )
            
            # คำนวณ Adjusted UPH
            self.clean_data['NO_BUMP'] = self.clean_data['NO_BUMP'].fillna(0)
            self.clean_data['NUMBER_REQUIRED'] = self.clean_data['NUMBER_REQUIRED'].fillna(0)
            
            # สูตรคำนวณ Adjusted UPH
            self.clean_data['adjusted_uph'] = self.clean_data['uph'] / (
                (self.clean_data['NO_BUMP'] / 2) + self.clean_data['NUMBER_REQUIRED']
            )
        else:
            print("ไม่มีข้อมูล Wire เสริม ใช้ UPH ดิบในการคำนวณ")
            self.clean_data['adjusted_uph'] = self.clean_data['uph']
        
        # คำนวณประสิทธิภาพสัมพัทธ์ (%)
        max_uph = self.clean_data.groupby('machine_model')['adjusted_uph'].max().to_dict()
        self.clean_data['efficiency'] = self.clean_data.apply(
            lambda row: (row['adjusted_uph'] / max_uph.get(row['machine_model'], 1)) * 100,
            axis=1
        )
    
    def detect_and_remove_outliers(self, method='auto'):
        """ตรวจจับและลบ Outliers ด้วยวิธีอัตโนมัติ"""
        if self.clean_data is None:
            print("ไม่มีข้อมูลที่ทำความสะอาดแล้ว")
            return False
        
        original_count = len(self.clean_data)
        self.clean_data['is_outlier'] = False
        
        # คำนวณ Outliers แยกตาม Machine Model
        for model in self.clean_data['machine_model'].unique():
            model_data = self.clean_data[self.clean_data['machine_model'] == model]
            uph_values = model_data['adjusted_uph']
            
            if method == 'auto':
                # ใช้ทั้ง Z-Score และ IQR
                z_scores = np.abs(stats.zscore(uph_values))
                iqr = stats.iqr(uph_values)
                q1, q3 = np.percentile(uph_values, [25, 75])
                
                # กำหนดเงื่อนไข Outlier
                outlier_condition = (
                    (z_scores > 3) | 
                    (uph_values < (q1 - 1.5 * iqr)) | 
                    (uph_values > (q3 + 1.5 * iqr)))
                
            elif method == 'iqr':
                # ใช้เฉพาะ IQR
                iqr = stats.iqr(uph_values)
                q1, q3 = np.percentile(uph_values, [25, 75])
                outlier_condition = (
                    (uph_values < (q1 - 1.5 * iqr)) | 
                    (uph_values > (q3 + 1.5 * iqr)))
            
            else:  # method == 'zscore'
                # ใช้เฉพาะ Z-Score
                z_scores = np.abs(stats.zscore(uph_values))
                outlier_condition = (z_scores > 3)
            
            # ทำเครื่องหมาย Outliers
            outlier_indices = model_data[outlier_condition].index
            self.clean_data.loc[outlier_indices, 'is_outlier'] = True
        
        # ลบ Outliers
        before_count = len(self.clean_data)
        self.clean_data = self.clean_data[~self.clean_data['is_outlier']]
        after_count = len(self.clean_data)
        
        print(f"ลบ Outliers สำเร็จ: ลบ {before_count - after_count} แถว (จากทั้งหมด {original_count} แถว)")
        return True
    
    def calculate_wb_stats(self):
        """คำนวณสถิติสำคัญสำหรับ Wire Bond"""
        if self.clean_data is None:
            print("ไม่มีข้อมูลที่ทำความสะอาดแล้ว")
            return False
        
        # สร้าง DataFrame สำหรับรายงาน
        stats_df = self.clean_data.groupby(['bom_no', 'machine_model']).agg({
            'adjusted_uph': ['mean', 'median', 'std', 'count'],
            'efficiency': 'mean',
            'wire_count': 'mean'
        }).reset_index()
        
        # จัดรูปแบบคอลัมน์ใหม่
        stats_df.columns = [
            'BOM', 'Machine Model', 
            'Mean Adjusted UPH', 'Median Adjusted UPH', 
            'Std Dev', 'Data Points',
            'Avg Efficiency (%)', 'Avg Wire Count'
        ]
        
        # คำนวณ Process Capability
        stats_df['Cp'] = (stats_df['Mean Adjusted UPH'] + 3*stats_df['Std Dev']) - (
                          stats_df['Mean Adjusted UPH'] - 3*stats_df['Std Dev']) / (6 * stats_df['Std Dev'])
        stats_df['Cpk'] = stats_df.apply(
            lambda row: min(
                (row['Mean Adjusted UPH'] - (row['Mean Adjusted UPH'] - 3*row['Std Dev'])) / (3 * row['Std Dev']),
                ((row['Mean Adjusted UPH'] + 3*row['Std Dev']) - row['Mean Adjusted UPH']) / (3 * row['Std Dev'])
            ),
            axis=1
        )
        
        self.final_report = stats_df.round(2)
        return True
    
    def generate_visualizations(self, output_folder='WB_Analysis_Results'):
        """สร้าง visualization และรายงาน"""
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        # 1. สร้างกราฟ UPH Trend
        self._plot_uph_trend(output_folder)
        
        # 2. สร้างกราฟประสิทธิภาพเครื่อง
        self._plot_machine_efficiency(output_folder)
        
        # 3. สร้างกราฟเปรียบเทียบ UPH ก่อน-หลังทำความสะอาด
        self._plot_uph_comparison(output_folder)
        
        # 4. บันทึกไฟล์ Excel
        self._save_excel_report(output_folder)
        
        print(f"\nสร้างรายงานและกราฟสำเร็จที่โฟลเดอร์: {output_folder}")
    
    def _plot_uph_trend(self, output_folder):
        """สร้างกราฟแสดงแนวโน้ม UPH"""
        plt.figure(figsize=(14, 7))
        
        # แยกสีตาม Machine Model
        unique_models = self.clean_data['machine_model'].unique()
        palette = sns.color_palette("husl", len(unique_models))
        
        for i, model in enumerate(unique_models):
            model_data = self.clean_data[self.clean_data['machine_model'] == model]
            plt.scatter(
                model_data['timestamp'], 
                model_data['adjusted_uph'], 
                color=palette[i],
                label=model,
                alpha=0.6
            )
            
            # เส้นแนวโน้ม
            sns.regplot(
                x=model_data['timestamp'].astype('int64')//10**9,
                y=model_data['adjusted_uph'],
                scatter=False,
                color=palette[i],
                line_kws={'linestyle':'--', 'alpha':0.5}
            )
        
        plt.title('Wire Bond Adjusted UPH Trend')
        plt.xlabel('Date/Time')
        plt.ylabel('Adjusted UPH')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()
        
        plot_path = os.path.join(output_folder, 'WB_UPH_Trend.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_machine_efficiency(self, output_folder):
        """สร้างกราฟประสิทธิภาพเครื่อง"""
        plt.figure(figsize=(12, 6))
        
        efficiency_data = self.clean_data.groupby('machine_model')['efficiency'].mean().sort_values()
        
        sns.barplot(
            x=efficiency_data.index,
            y=efficiency_data.values,
            palette='Blues_d'
        )
        
        plt.title('Average Machine Efficiency (%)')
        plt.xlabel('Machine Model')
        plt.ylabel('Efficiency (%)')
        plt.xticks(rotation=45)
        plt.ylim(0, 100)
        
        # เพิ่มค่าเปอร์เซ็นต์บนกราฟ
        for i, v in enumerate(efficiency_data.values):
            plt.text(i, v + 2, f"{v:.1f}%", ha='center')
        
        plot_path = os.path.join(output_folder, 'WB_Machine_Efficiency.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_uph_comparison(self, output_folder):
        """สร้างกราฟเปรียบเทียบ UPH ก่อน-หลังทำความสะอาด"""
        plt.figure(figsize=(14, 6))
        
        plt.subplot(1, 2, 1)
        sns.boxplot(
            x=self.raw_data['machine_model'],
            y=self.raw_data['uph'],
            palette='Oranges'
        )
        plt.title('Original UPH (Before Cleaning)')
        plt.xlabel('Machine Model')
        plt.ylabel('UPH')
        plt.xticks(rotation=45)
        
        plt.subplot(1, 2, 2)
        sns.boxplot(
            x=self.clean_data['machine_model'],
            y=self.clean_data['adjusted_uph'],
            palette='Blues'
        )
        plt.title('Adjusted UPH (After Cleaning)')
        plt.xlabel('Machine Model')
        plt.ylabel('Adjusted UPH')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        plot_path = os.path.join(output_folder, 'WB_UPH_Comparison.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _save_excel_report(self, output_folder):
        """บันทึกผลลัพธ์เป็นไฟล์ Excel"""
        report_path = os.path.join(output_folder, 'WB_UPH_Analysis_Report.xlsx')
        
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
    analyzer = WB_UPH_Analyzer()
    
    # โหลดข้อมูลการผลิต WB (เปลี่ยน path เป็นของไฟล์คุณ)
    production_file = "wb_production_data.xlsx"
    if not analyzer.load_production_data(production_file):
        exit()
    
    # โหลดข้อมูล Wire เสริม (optional)
    wire_file = "wire_data.xlsx"
    if os.path.exists(wire_file):
        analyzer.load_wire_data(wire_file)
    
    # ประมวลผลข้อมูล
    if analyzer.preprocess_data():
        # ตรวจจับและลบ Outliers
        analyzer.detect_and_remove_outliers(method='auto')
        
        # คำนวณสถิติ
        analyzer.calculate_wb_stats()
        
        # สร้างรายงานและกราฟ
        analyzer.generate_visualizations()
        
        # แสดงตัวอย่างผลลัพธ์
        print("\nตัวอย่างรายงานสรุป:")
        print(analyzer.final_report.head())