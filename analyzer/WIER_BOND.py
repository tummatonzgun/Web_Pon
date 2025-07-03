import pandas as pd
import numpy as np
from scipy.stats import zscore
import matplotlib.pyplot as plt
import os

class WireBondingAnalyzer:
    def __init__(self):
        self.nobump_df = None
        self.wb_data = None
        self.efficiency_df = None
    
    def load_data(self, uph_path, wire_data_path):
        """โหลดข้อมูลที่จำเป็น"""
        try:
            # โหลดข้อมูล Wire Data
            self.nobump_df = pd.read_excel(wire_data_path)
            self.nobump_df.columns = self.nobump_df.columns.str.upper()  # เปลี่ยนเป็นตัวพิมพ์ใหญ่ทั้งหมด
            
            # โหลดข้อมูล UPH
            self.wb_data = pd.read_excel(uph_path)
            print("✅ โหลดข้อมูลสำเร็จแล้ว")
            return True
        except Exception as e:
            print(f"❌ เกิดข้อผิดพลาดขณะโหลดข้อมูล: {e}")
            return False
    
    def calculate_wire_per_unit(self, bom_no):
        """คำนวณจำนวนสายต่อหน่วย"""
        try:
            bom_data = self.nobump_df[self.nobump_df['BOM_NO'] == bom_no]
            
            no_bump = float(bom_data['NO_BUMP'].iloc[0]) if not bom_data['NO_BUMP'].empty else 0
            num_required = float(bom_data['NUMBER_REQUIRED'].iloc[0]) if 'NUMBER_REQUIRED' in bom_data.columns and not bom_data['NUMBER_REQUIRED'].empty else 0
            
            wire_per_unit = (no_bump / 2) + num_required
            return wire_per_unit if wire_per_unit > 0 else 1
        except:
            return 1
    
    def remove_outliers(self, df):
        """ลบ outliers จากข้อมูล"""
        try:
            df.columns = df.columns.str.lower()
            if 'uph' not in df.columns or 'machine model' not in df.columns:
                raise KeyError("Missing required columns")
            
            models = df['machine model'].unique()
            cleaned_data = []
            
            for model in models:
                model_data = df[df['machine model'] == model].copy()
                
                # ใช้ Z-Score (±3 SD)
                z_threshold = 3
                z_scores = zscore(model_data['uph'])
                model_data = model_data[(z_scores >= -z_threshold) & (z_scores <= z_threshold)]
                
                # ใช้ IQR (1.5 * IQR)
                Q1 = model_data['uph'].quantile(0.25)
                Q3 = model_data['uph'].quantile(0.75)
                IQR = Q3 - Q1
                model_data = model_data[
                    (model_data['uph'] >= Q1 - 1.5*IQR) & 
                    (model_data['uph'] <= Q3 + 1.5*IQR)]
                
                model_data['outlier_method'] = 'Z-Score + IQR'
                cleaned_data.append(model_data)
            
            return pd.concat(cleaned_data) if cleaned_data else df
        except Exception as e:
            print(f"Error removing outliers: {e}")
            return df
    
    def calculate_efficiency(self):
        """คำนวณประสิทธิภาพการทำงาน"""
        if self.wb_data is None or self.nobump_df is None:
            print("❌ โปรดโหลดข้อมูลก่อนทำการคำนวณ")
            return None
        
        try:
            # ทำความสะอาดข้อมูล
            cleaned_data = self.remove_outliers(self.wb_data)
            
            # ตรวจสอบว่ามีข้อมูลหลังจากทำความสะอาดแล้ว
            if cleaned_data.empty:
                print("⚠️ ไม่มีข้อมูลเหลือหลังจากการลบ outliers")
                return None
            
            # กลุ่มข้อมูลตาม BOM และรุ่นเครื่อง
            grouped = cleaned_data.groupby(['bom_no', 'machine model'])
            results = []
            
            for (bom_no, model), group in grouped:
                mean_uph = group['uph'].mean()
                wire_per_unit = self.calculate_wire_per_unit(bom_no)
                efficiency = mean_uph / wire_per_unit if wire_per_unit > 0 else 0
                
                results.append({
                    'BOM': bom_no,
                    'Model': model,
                    'Mean_UPH': round(mean_uph, 2),
                    'Wire_Per_Unit': round(wire_per_unit, 2),
                    'Efficiency': round(efficiency, 3),
                    'Data_Points': len(group),
                    'Outlier_Method': group['outlier_method'].iloc[0] if 'outlier_method' in group.columns else 'N/A'
                })
            
            self.efficiency_df = pd.DataFrame(results)
            print("✅ คำนวณประสิทธิภาพเสร็จสิ้น")
            return self.efficiency_df
        except Exception as e:
            print(f"❌ เกิดข้อผิดพลาดขณะคำนวณประสิทธิภาพ: {e}")
            return None
    
    def generate_report(self):
        """สร้างรายงานสรุป"""
        if self.efficiency_df is None:
            print("❌ โปรดคำนวณประสิทธิภาพก่อนสร้างรายงาน")
            return None
        
        try:
            report = {
                'summary': {
                    'average_efficiency': round(self.efficiency_df['Efficiency'].mean(), 3),
                    'best_model': self.efficiency_df.loc[self.efficiency_df['Efficiency'].idxmax()]['Model'],
                    'best_efficiency': round(self.efficiency_df['Efficiency'].max(), 3),
                    'worst_model': self.efficiency_df.loc[self.efficiency_df['Efficiency'].idxmin()]['Model'],
                    'worst_efficiency': round(self.efficiency_df['Efficiency'].min(), 3),
                    'total_boms': len(self.efficiency_df['BOM'].unique()),
                    'total_models': len(self.efficiency_df['Model'].unique())
                },
                'details': self.efficiency_df.to_dict('records')
            }
            
            print("✅ สร้างรายงานเสร็จสิ้น")
            return report
        except Exception as e:
            print(f"❌ เกิดข้อผิดพลาดขณะสร้างรายงาน: {e}")
            return None
    
    def plot_results(self, save_path='./wb_analysis_results.png'):
        """สร้างกราฟแสดงผล"""
        if self.efficiency_df is None:
            print("❌ โปรดคำนวณประสิทธิภาพก่อนสร้างกราฟ")
            return False
        
        try:
            plt.figure(figsize=(15, 6))
            
            # กราฟแท่งประสิทธิภาพตามรุ่นเครื่อง
            plt.subplot(1, 2, 1)
            self.efficiency_df.groupby('Model')['Efficiency'].mean().sort_values().plot(
                kind='barh', color='skyblue')
            plt.title('ประสิทธิภาพเฉลี่ยตามรุ่นเครื่อง', pad=20)
            plt.xlabel('อัตราส่วนประสิทธิภาพ')
            plt.grid(axis='x', linestyle='--', alpha=0.7)
            
            # กราฟกระจายความสัมพันธ์ UPH และประสิทธิภาพ
            plt.subplot(1, 2, 2)
            plt.scatter(
                self.efficiency_df['Mean_UPH'], 
                self.efficiency_df['Efficiency'],
                c='green', alpha=0.6, edgecolors='w', s=100)
            
            # เพิ่มเส้นแนวโน้ม
            z = np.polyfit(self.efficiency_df['Mean_UPH'], self.efficiency_df['Efficiency'], 1)
            p = np.poly1d(z)
            plt.plot(
                self.efficiency_df['Mean_UPH'], 
                p(self.efficiency_df['Mean_UPH']), 
                "r--", linewidth=2)
            
            plt.title('ความสัมพันธ์ระหว่าง UPH และประสิทธิภาพ', pad=20)
            plt.xlabel('UPH เฉลี่ย')
            plt.ylabel('อัตราส่วนประสิทธิภาพ')
            plt.grid(True, linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            
            # สร้างโฟลเดอร์ถ้ายังไม่มี
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ สร้างกราฟเสร็จสิ้นและบันทึกที่ {save_path}")
            return True
        except Exception as e:
            print(f"❌ เกิดข้อผิดพลาดขณะสร้างกราฟ: {e}")
            return False
    
    def export_to_excel(self, file_path='./wb_analysis_results.xlsx'):
        """ส่งออกผลลัพธ์เป็นไฟล์ Excel"""
        if self.efficiency_df is None:
            print("❌ โปรดคำนวณประสิทธิภาพก่อนส่งออกไฟล์")
            return False
        
        try:
            # สร้าง Excel writer
            writer = pd.ExcelWriter(file_path, engine='xlsxwriter')
            
            # Sheet 1: ผลลัพธ์ประสิทธิภาพ
            self.efficiency_df.to_excel(
                writer, sheet_name='Efficiency_Results', index=False)
            
            # Sheet 2: สรุปตามรุ่นเครื่อง
            model_summary = self.efficiency_df.groupby('Model').agg({
                'Efficiency': ['mean', 'std', 'count'],
                'Mean_UPH': 'mean'
            }).round(3)
            model_summary.to_excel(writer, sheet_name='Model_Summary')
            
            # ปิดและบันทึกไฟล์
            writer.close()
            
            print(f"✅ ส่งออกไฟล์ Excel เสร็จสิ้นที่ {file_path}")
            return True
        except Exception as e:
            print(f"❌ เกิดข้อผิดพลาดขณะส่งออกไฟล์ Excel: {e}")
            return False

# ตัวอย่างการใช้งาน
if __name__ == "__main__":
    analyzer = WireBondingAnalyzer()
    
    # โหลดข้อมูล (เปลี่ยน path ตามจริง)
    analyzer.load_data(
        uph_path='./data/Data WB UTL1 Jan-May-25.xlsx',
        wire_data_path='./temp/Book6_Wire Data.xlsx'
    )
    
    # คำนวณประสิทธิภาพ
    efficiency_df = analyzer.calculate_efficiency()
    
    if efficiency_df is not None:
        # แสดงผลลัพธ์ตัวอย่าง
        print("\nตัวอย่างผลลัพธ์:")
        print(efficiency_df.head())
        
        # สร้างรายงาน
        report = analyzer.generate_report()
        if report:
            print("\nสรุปผลการวิเคราะห์:")
            print(f"ประสิทธิภาพเฉลี่ย: {report['summary']['average_efficiency']}")
            print(f"รุ่นเครื่องที่ดีที่สุด: {report['summary']['best_model']} (ประสิทธิภาพ: {report['summary']['best_efficiency']})")
        
        # สร้างกราฟและส่งออกไฟล์
        analyzer.plot_results()
        analyzer.export_to_excel()