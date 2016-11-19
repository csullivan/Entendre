#include <algorithm>
#include <iostream>
#include <unordered_map>
#include <vector>

template<class Key, class T>
class insertion_ordered_map : public std::unordered_map<Key,T> {
public:
  //using std::unordered_map<Key,T>::find;
  using std::unordered_map<Key,T>::begin;
  using std::unordered_map<Key,T>::end;

  auto find(const Key& key) const { return std::unordered_map<Key,T>::find(key); }
  auto operator[](std::size_t pos) const { return std::pair<Key,T>(insertion_order[pos],std::unordered_map<Key,T>::at(insertion_order[pos])); }
  auto size() const { return insertion_order.size(); }

  std::pair<typename std::unordered_map<Key,T>::iterator,bool> insert( const std::pair<const Key, T>& value ) {
    auto result = std::unordered_map<Key,T>::insert(value);
    if (result.second) { insertion_order.push_back(value.first); }
    return result;
  }
  std::pair<typename std::unordered_map<Key,T>::iterator,bool> insert( const std::pair<const Key, T>&& value ) {
    auto result = std::unordered_map<Key,T>::insert(value);
    if (result.second) { insertion_order.push_back(value.first); }
    return result;
  }

private:
  std::vector<Key> insertion_order;
};
