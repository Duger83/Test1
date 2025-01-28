import pytest

class BooksCollector:
    
    def __init__(self):
        self.books_genre = {}
        self.favorites = []
        self.genre = []
        self.genre_age_rating = {}

    def add_new_book(self, book_name):
        self.books_genre[book_name] = None

    def set_book_genre(self, book_name, genre):
        if book_name in self.books_genre:
            self.books_genre[book_name] = genre
            if genre not in self.genre:
                self.genre.append(genre)
    
    def set_genre_age_rating(self, genre, rating):
        if genre not in self.genre:
            self.genre.append(genre)
        self.genre_age_rating[genre] = rating

    def get_book_genre(self, book_name):
        return self.books_genre.get(book_name, "Книга не найдена")

    def get_books_with_specific_genre(self, genre):
        return [book for book, g in self.books_genre.items() if g == genre]

    def get_books_genre(self):
        return self.books_genre

    def get_books_for_children(self):
        return [book for book, g in self.books_genre.items() if g in self.genre_age_rating and self.genre_age_rating[g] <= 7]

    def add_book_in_favorites(self, book_name):
        if book_name in self.books_genre and book_name not in self.favorites:
            self.favorites.append(book_name)

    def delete_book_from_favorites(self, book_name):
        if book_name in self.favorites:
            self.favorites.remove(book_name)

    def get_list_of_favorites_books(self):
        return self.favorites

collector = BooksCollector()
collector.add_new_book('Властелин колец')
collector.add_new_book('Гарри Поттер')
collector.add_new_book('Матрица')
collector.set_book_genre('Властелин колец', 'Фантастика')
collector.set_book_genre('Гарри Поттер', 'Фэнтези')
collector.set_book_genre('Матрица', 'Научная фантастика')
print(collector.get_book_genre('Властелин колец'))
print(collector.get_book_genre('Гарри Поттер')) 
print(collector.get_book_genre('Матрица')) 
fantasy_books = collector.get_books_with_specific_genre('Фантастика')
print(f'Книги в жанре Фантастика: {fantasy_books}')
all_books = collector.get_books_genre()
print('Все книги и их жанры:', all_books)
collector.set_genre_age_rating('Фантастика', 16)
collector.set_genre_age_rating('Фэнтези', 6)
collector.set_genre_age_rating('Научная фантастика', 12)
children_books = collector.get_books_for_children()
print('Книги для детей:', children_books)
collector.add_book_in_favorites('Властелин колец')
collector.add_book_in_favorites('Гарри Поттер')
collector.delete_book_from_favorites('Гарри Поттер')
favorites_list = collector.get_list_of_favorites_books()
print('Список избранных книг:', favorites_list)

def test_add_new_book():
    collector = BooksCollector()
    collector.add_new_book('Колобок')
    assert "Колобок" in collector.books_genre # успешное добавление книги
    assert "Колосок" in collector.books_genre, 'Книга не добавлена в словарь' # негативный случай

def test_set_book_genre():
    collector = BooksCollector()
    collector.add_new_book('Колосок')
    collector.set_book_genre('Колосок', 'Сказки')
    assert collector.get_book_genre('Колосок') == 'Сказки' # успешное добавление жанра для книги
    assert collector.get_book_genre('Колобок') == 'Сказки', 'Ошибка жанра, или жанр книги не установлен' # негативный случай

def test_get_books_with_specific_genre():
    collector = BooksCollector()
    collector.add_new_book('Дневник неудачника')
    collector.set_book_genre('Дневник неудачника', 'Фикшн')
    collector.add_new_book('Черная обезъяна')
    collector.set_book_genre('Черная обезьяна', 'Нон-Фикшн')
    collector.add_new_book('Собаки и другие люди')
    collector.set_book_genre('Собаки и другие люди', 'Фикшн')
    assert collector.get_books_with_specific_genre("Фикшн") == ['Дневник неудачника', 'Собаки и другие люди'] # успешный поиск книг по заданному жанру
    assert collector.get_books_with_specific_genre("Фикшн") == ['Бег'], 'Неверно построен список книг заданного жанра' # негативный случай

def test_get_books_genre():
    collector = BooksCollector()
    collector.add_new_book('Ветер')
    collector.set_book_genre('Ветер', 'Фикшн')
    assert collector.get_books_genre() == {'Ветер': 'Фикшн'} # успешное вывод жанра книги
    assert collector.get_books_genre() == {'Ветер': 'Нон-Фикшн'}, 'несоответствие названия книги и жанра' # негативный случай

def test_get_books_for_children():
    collector = BooksCollector()
    collector.genre_age_rating = {'Комикс': 5}
    collector.add_new_book('Человек-паук')
    collector.set_book_genre('Человек-паук', 'Комикс')
    collector.add_new_book('Игрок')
    collector.set_book_genre('Игрок', 'Роман')
    assert collector.get_books_for_children() == ['Человек-паук'] # успешно найдена детская книга
    assert collector.get_books_for_children() == ['Игрок'], 'найдена не детская книга' # негативный случай

def test_add_book_in_favorites():
    collector = BooksCollector()
    collector.add_new_book('Идиот')
    collector.add_book_in_favorites('Идиот')
    assert 'Идиот' in collector.favorites # книга успешно добавлена в избранное
    assert 'Идиот' not in collector.favorites, 'ошибка добавления книги в список избранных' # негативный случай

def test_delete_book_from_favorites():
    collector = BooksCollector()
    collector.delete_book_from_favorites('Идиот')
    assert 'Идиот' not in collector.favorites # книга успешно удалена из избранного
    assert 'Идиот' in collector.favorites, 'ошибка удаления книги из списка избранных' # негативный случай

def test_get_list_of_favorites_books():
    collector = BooksCollector()
    collector.add_new_book('Бег')
    collector.add_book_in_favorites('Бег')
    assert collector.get_list_of_favorites_books() == ['Бег'] # успешное сформирован список избранных книг
    assert collector.get_list_of_favorites_books() == ['Собачье сердце'], 'ошибка получения списка избранных книг' # негативный случай

#test_get_books_with_specific_genre()
#test_get_books_genre()
#test_get_list_of_favorites_books()
#test_delete_book_from_favorites()
#test_add_book_in_favorites()
test_get_books_for_children() #тестируем метод поиска детский книг
#test_get_books_genre()
#test_get_books_with_specific_genre()
#test_set_book_genre()
#test_add_new_book()