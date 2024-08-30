#include <iostream>
#include <unordered_map>
using namespace std;

int main() {
    int n;
    cin >> n; 
    string str;
    cin >> str;
    char vowels[] = {'a', 'e', 'i', 'o', 'u'};
    
    unordered_map<char, int> vcount;
    
    for (char vowel : vowels) {
        vcount[vowel] = 0;
    }
    
    for (char c : str) {
        if (vcount.find(c) != vcount.end()) {
            vcount[c]++;
        }
    }
    
    char lastVowel = '\0';
    int maxCount = 0;
    
    for (char vowel : vowels) {
        if (vcount[vowel] > maxCount) {
            maxCount = vcount[vowel];
            lastVowel = vowel;
        }
    }
    
    cout << lastVowel << endl;
    
    return 0;
}
