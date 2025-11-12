import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import string

class TextQualityAnalyzer:
    def __init__(self):
        self.setup_fuzzy_system()
    
    def setup_fuzzy_system(self):
        # Створюємо антецеденти (вхідні змінні)
        self.spelling = ctrl.Antecedent(np.arange(0, 101, 1), 'spelling')
        self.punctuation = ctrl.Antecedent(np.arange(0, 101, 1), 'punctuation')
        self.vocabulary = ctrl.Antecedent(np.arange(0, 101, 1), 'vocabulary')
        self.redundancy = ctrl.Antecedent(np.arange(0, 101, 1), 'redundancy')
        
        # Створюємо консеквент (вихідна змінна)
        self.quality = ctrl.Consequent(np.arange(0, 101, 1), 'quality')
        
        # Власні функції належності для більш точної роботи
        # Орфографія
        self.spelling['погана'] = fuzz.trimf(self.spelling.universe, [0, 0, 50])
        self.spelling['середня'] = fuzz.trimf(self.spelling.universe, [30, 50, 70])
        self.spelling['хороша'] = fuzz.trimf(self.spelling.universe, [50, 100, 100])
        
        # Пунктуація
        self.punctuation['погана'] = fuzz.trimf(self.punctuation.universe, [0, 0, 40])
        self.punctuation['середня'] = fuzz.trimf(self.punctuation.universe, [30, 50, 70])
        self.punctuation['хороша'] = fuzz.trimf(self.punctuation.universe, [60, 100, 100])
        
        # Словниковий запас
        self.vocabulary['поганий'] = fuzz.trimf(self.vocabulary.universe, [0, 0, 40])
        self.vocabulary['середній'] = fuzz.trimf(self.vocabulary.universe, [30, 50, 70])
        self.vocabulary['хороший'] = fuzz.trimf(self.vocabulary.universe, [60, 100, 100])
        
        # Надмірність
        self.redundancy['низька'] = fuzz.trimf(self.redundancy.universe, [0, 0, 30])
        self.redundancy['середня'] = fuzz.trimf(self.redundancy.universe, [20, 50, 80])
        self.redundancy['висока'] = fuzz.trimf(self.redundancy.universe, [70, 100, 100])
        
        # Якість тексту
        self.quality['погана'] = fuzz.trimf(self.quality.universe, [0, 0, 40])
        self.quality['середня'] = fuzz.trimf(self.quality.universe, [30, 50, 70])
        self.quality['хороша'] = fuzz.trimf(self.quality.universe, [60, 100, 100])
        
        # Правила нечіткої системи
        self.rules = [
            # Критичні правила
            ctrl.Rule(self.spelling['погана'] | self.punctuation['погана'], self.quality['погана']),
            ctrl.Rule(self.redundancy['висока'], self.quality['погана']),
            ctrl.Rule(self.vocabulary['поганий'], self.quality['погана']),
            
            # Позитивні правила
            ctrl.Rule(self.spelling['хороша'] & self.punctuation['хороша'], self.quality['хороша']),
            ctrl.Rule(self.vocabulary['хороший'] & self.redundancy['низька'], self.quality['хороша']),
            
            # Комбіновані правила
            ctrl.Rule(self.spelling['середня'] & self.punctuation['середня'], self.quality['середня']),
            ctrl.Rule(self.spelling['хороша'] & self.punctuation['хороша'] & 
                     self.vocabulary['хороший'] & self.redundancy['низька'], self.quality['хороша'])
        ]
        
        # Створюємо систему контролю
        self.quality_ctrl = ctrl.ControlSystem(self.rules)
        self.quality_sim = ctrl.ControlSystemSimulation(self.quality_ctrl)
    
    def analyze_text(self, text):
        """Аналіз тексту та повернення оцінок за критеріями"""
        words = text.lower().split()
        total_words = len(words)
        
        if total_words == 0:
            return {
                'spelling': 0,
                'punctuation': 0,
                'vocabulary': 0,
                'redundancy': 100,
                'quality': 0
            }
        
        # Перевірка орфографії
        spelling_score = self.check_spelling_simple(words)
        
        # Перевірка пунктуації
        punctuation_score = self.check_punctuation(text)
        
        # Оцінка словникового запасу
        vocabulary_score = self.check_vocabulary(words)
        
        # Перевірка на надмірність
        redundancy_score = self.check_redundancy(words)
        
        # Використання нечіткої логіки
        self.quality_sim.input['spelling'] = spelling_score
        self.quality_sim.input['punctuation'] = punctuation_score
        self.quality_sim.input['vocabulary'] = vocabulary_score
        self.quality_sim.input['redundancy'] = redundancy_score
        
        try:
            self.quality_sim.compute()
            quality_score = self.quality_sim.output['quality']
        except:
            # Резервний розрахунок
            quality_score = (spelling_score + punctuation_score + vocabulary_score + (100 - redundancy_score)) / 4
        
        return {
            'spelling': spelling_score,
            'punctuation': punctuation_score,
            'vocabulary': vocabulary_score,
            'redundancy': redundancy_score,
            'quality': quality_score
        }
    
    def check_spelling_simple(self, words):
        """Спрощена перевірка орфографії українською"""
        ukrainian_words = {
            'це', 'так', 'не', 'що', 'як', 'ми', 'ви', 'вони', 'він', 'вона', 'воно',
            'для', 'про', 'над', 'під', 'перед', 'після', 'через', 'в', 'на', 'за',
            'дуже', 'трохи', 'багато', 'мало', 'швидко', 'повільно', 'гарний', 'добрий',
            'великий', 'малий', 'новий', 'старий', 'час', 'місце', 'людина', 'робота',
            'привіт', 'світ', 'як', 'справи', 'добре', 'дякую', 'текст', 'слово', 'речення'
        }
        
        correct_count = sum(1 for word in words if word in ukrainian_words)
        return (correct_count / len(words)) * 100 if words else 0
    
    def check_punctuation(self, text):
        """Перевірка пунктуації"""
        if len(text.strip()) == 0:
            return 0
            
        # Перевірка наявності розділових знаків
        punctuation_marks = sum(1 for char in text if char in '.,!?;:')
        words_count = len(text.split())
        
        if words_count == 0:
            return 0
            
        punctuation_ratio = punctuation_marks / words_count
        
        # Нормалізація до 100 балів
        if punctuation_ratio > 0.3:
            return 70
        elif punctuation_ratio > 0.1:
            return 90
        elif punctuation_ratio > 0.05:
            return 60
        else:
            return 30
    
    def check_vocabulary(self, words):
        """Оцінка словникового запасу"""
        if not words:
            return 0
        
        # Різноманітність слів
        unique_words = set(words)
        diversity_ratio = len(unique_words) / len(words)
        
        # Складність слів (довжина)
        complex_words = sum(1 for word in words if len(word) > 6)
        complexity_ratio = complex_words / len(words)
        
        # Комбінована оцінка
        return (diversity_ratio * 0.7 + complexity_ratio * 0.3) * 100
    
    def check_redundancy(self, words):
        """Перевірка на надмірність"""
        if len(words) < 3:
            return 0
        
        from collections import Counter
        word_counts = Counter(words)
        
        # Знаходимо слова, що повторюються
        total_repeats = sum(count - 1 for count in word_counts.values() if count > 1)
        
        if len(words) == 0:
            return 0
            
        redundancy_ratio = total_repeats / len(words)
        return min(redundancy_ratio * 150, 100)
    
    def visualize_all_membership_functions(self):
        """Візуалізація всіх функцій належності в одному вікні"""
        fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(15, 12))
        
        # Орфографія
        ax1.plot(self.spelling.universe, fuzz.trimf(self.spelling.universe, [0, 0, 50]), 'b', linewidth=1.5, label='Погана')
        ax1.plot(self.spelling.universe, fuzz.trimf(self.spelling.universe, [30, 50, 70]), 'g', linewidth=1.5, label='Середня')
        ax1.plot(self.spelling.universe, fuzz.trimf(self.spelling.universe, [50, 100, 100]), 'r', linewidth=1.5, label='Хороша')
        ax1.set_title('Функції належності: Орфографія', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Пунктуація
        ax2.plot(self.punctuation.universe, fuzz.trimf(self.punctuation.universe, [0, 0, 40]), 'b', linewidth=1.5, label='Погана')
        ax2.plot(self.punctuation.universe, fuzz.trimf(self.punctuation.universe, [30, 50, 70]), 'g', linewidth=1.5, label='Середня')
        ax2.plot(self.punctuation.universe, fuzz.trimf(self.punctuation.universe, [60, 100, 100]), 'r', linewidth=1.5, label='Хороша')
        ax2.set_title('Функції належності: Пунктуація', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Словниковий запас
        ax3.plot(self.vocabulary.universe, fuzz.trimf(self.vocabulary.universe, [0, 0, 40]), 'b', linewidth=1.5, label='Поганий')
        ax3.plot(self.vocabulary.universe, fuzz.trimf(self.vocabulary.universe, [30, 50, 70]), 'g', linewidth=1.5, label='Середній')
        ax3.plot(self.vocabulary.universe, fuzz.trimf(self.vocabulary.universe, [60, 100, 100]), 'r', linewidth=1.5, label='Хороший')
        ax3.set_title('Функції належності: Словниковий запас', fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Надмірність
        ax4.plot(self.redundancy.universe, fuzz.trimf(self.redundancy.universe, [0, 0, 30]), 'g', linewidth=1.5, label='Низька')
        ax4.plot(self.redundancy.universe, fuzz.trimf(self.redundancy.universe, [20, 50, 80]), 'y', linewidth=1.5, label='Середня')
        ax4.plot(self.redundancy.universe, fuzz.trimf(self.redundancy.universe, [70, 100, 100]), 'r', linewidth=1.5, label='Висока')
        ax4.set_title('Функції належності: Надмірність', fontsize=12, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Якість тексту
        ax5.plot(self.quality.universe, fuzz.trimf(self.quality.universe, [0, 0, 40]), 'b', linewidth=1.5, label='Погана')
        ax5.plot(self.quality.universe, fuzz.trimf(self.quality.universe, [30, 50, 70]), 'g', linewidth=1.5, label='Середня')
        ax5.plot(self.quality.universe, fuzz.trimf(self.quality.universe, [60, 100, 100]), 'r', linewidth=1.5, label='Хороша')
        ax5.set_title('Функції належності: Якість тексту', fontsize=12, fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Пустий простір замість тексту
        ax6.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def plot_combined_surface(self):
        """Комбінована поверхня відгуку"""
        fig = plt.figure(figsize=(14, 6))
        
        # Поверхня 1: Орфографія vs Пунктуація
        ax1 = fig.add_subplot(121, projection='3d')
        x = np.arange(0, 101, 5)
        y = np.arange(0, 101, 5)
        X, Y = np.meshgrid(x, y)
        Z1 = np.zeros(X.shape)
        
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                try:
                    self.quality_sim.input['spelling'] = X[i, j]
                    self.quality_sim.input['punctuation'] = Y[i, j]
                    self.quality_sim.input['vocabulary'] = 50
                    self.quality_sim.input['redundancy'] = 30
                    self.quality_sim.compute()
                    Z1[i, j] = self.quality_sim.output['quality']
                except:
                    Z1[i, j] = (X[i, j] + Y[i, j]) / 2
        
        surf1 = ax1.plot_surface(X, Y, Z1, cmap='viridis', alpha=0.8)
        ax1.set_xlabel('Орфографія')
        ax1.set_ylabel('Пунктуація')
        ax1.set_zlabel('Якість')
        ax1.set_title('Вплив орфографії та пунктуації на якість тексту')
        
        # Поверхня 2: Словниковий запас vs Надмірність
        ax2 = fig.add_subplot(122, projection='3d')
        Z2 = np.zeros(X.shape)
        
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                try:
                    self.quality_sim.input['spelling'] = 70
                    self.quality_sim.input['punctuation'] = 70
                    self.quality_sim.input['vocabulary'] = X[i, j]
                    self.quality_sim.input['redundancy'] = Y[i, j]
                    self.quality_sim.compute()
                    Z2[i, j] = self.quality_sim.output['quality']
                except:
                    Z2[i, j] = (X[i, j] + (100 - Y[i, j])) / 2
        
        surf2 = ax2.plot_surface(X, Y, Z2, cmap='plasma', alpha=0.8)
        ax2.set_xlabel('Словниковий запас')
        ax2.set_ylabel('Надмірність')
        ax2.set_zlabel('Якість')
        ax2.set_title('Вплив словникового запасу та надмірності на якість тексту')
        
        plt.tight_layout()
        plt.show()

# Демонстрація роботи системи
def main():
    analyzer = TextQualityAnalyzer()
    
    print("=== СИСТЕМА АНАЛІЗУ ЯКОСТІ ТЕКСТУ ===")
    print("Реалізація з використанням нечіткої логіки (Mamdani)\n")
    
    # Українські тестові тексти
    test_texts = [
        "Це чудовий приклад якісного тексту. Він має правильну пунктуацію, гарну орфографію та різноманітний словниковий запас. Кожне речення починається з великої літери та закінчується крапкою.",
        "привіт світ як справи я добре дякую а ти цей текст не має пунктуації та використовує дуже прості слова повторення слів слова слова",
        "Ерудований, вишуканий та екстравагантний словниковий запас демонструє неперевершену лінгвістичну майстерність та всеосяжне мовне багатство через складну фразеологію.",
        "слово слово слово повторення повторення повторення те саме те саме нудно нудно текст текст текст багато разів багато разів"
    ]
    
    descriptions = [
        "Якісний текст з правильною пунктуацією",
        "Простий текст без пунктуації з повторами",
        "Текст з багатим словниковим запасом", 
        "Текст з високою надмірністю"
    ]
    
    print("РЕЗУЛЬТАТИ ТЕСТУВАННЯ:\n")
    
    for i, (text, desc) in enumerate(zip(test_texts, descriptions), 1):
        print(f"--- ТЕСТ {i}: {desc} ---")
        print(f"Текст: {text[:80]}...")
        
        results = analyzer.analyze_text(text)
        
        print("\nРезультати аналізу:")
        print(f"Орфографія: {results['spelling']:.1f}%")
        print(f"Пунктуація: {results['punctuation']:.1f}%")
        print(f"Словниковий запас: {results['vocabulary']:.1f}%")
        print(f"Надмірність: {results['redundancy']:.1f}%")
        print(f"Загальна якість: {results['quality']:.1f}%")
        
        # Інтерпретація результату
        if results['quality'] >= 80:
            assessment = "ВІДМІННО"
        elif results['quality'] >= 60:
            assessment = "ДОБРЕ"
        elif results['quality'] >= 40:
            assessment = "ЗАДОВІЛЬНО"
        else:
            assessment = "ПОГАНО"
        
        print(f"ОЦІНКА: {assessment}\n")
        print("-" * 50)
    
    # Візуалізація
    print("\n=== ВІЗУАЛІЗАЦІЯ РЕЗУЛЬТАТІВ ===")
    analyzer.visualize_all_membership_functions()
    analyzer.plot_combined_surface()

if __name__ == "__main__":
    main()