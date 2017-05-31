#ifndef _HASH_H_
#define _HASH_H_

template<unsigned long... primes>
struct Hash_Impl;

/// Base case, if we've run out of prime numbers.
template<>
struct Hash_Impl<> {
  static constexpr unsigned long hash_impl() {
    return 0;
  }
};

template<unsigned long first, unsigned long... rest>
struct Hash_Impl<first, rest...> {
  /// Peel off one argument, multiply it by a prime number, and continue.
  template<typename... T>
  static constexpr unsigned long hash_impl(unsigned long a, T... b) {
    return a*first + Hash_Impl<rest...>::hash_impl(b...);
  }

  /// All arguments are exhausted, no more needed.
  static constexpr unsigned long hash_impl() {
    return 0;
  }

  /** Given a list of unsigned long, hash them into a single unsigned
   * long.
   *
   * Each input value is multiplied by a prime number, then they are
   * all summed together.  We add one prime number to avoid repetition
   * with repeated application of the hash. With unsigned 64-bit
   * arithmetic, it wont repeat until n*prime % 2^64 == 1.
   */
  template<typename... T>
  static constexpr unsigned long hash(T... a) {
    static_assert(sizeof...(rest) >= sizeof...(T),
                  "Not enough prime numbers to hash that many arguments");
    return first + Hash_Impl<rest...>::hash_impl((unsigned long)a...);
  }
};

/// Lots of prime numbers.  If the static assert above gets triggers,
/// add some more.
typedef Hash_Impl<
  5915587277UL,
  1500450271UL,
  3267000013UL,
  5754853343UL,
  4093082899UL,
  9576890767UL,
  3628273133UL,
  2860486313UL,
  5463458053UL,
  3367900313UL> Hasher;



#endif /* _HASH_H_ */
