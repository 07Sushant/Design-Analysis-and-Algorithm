#include <iostream>
#include <vector>
#include <string>
#include <array>
#include <cassert>

using namespace std;

class Node
{
public:
    static const int Letters = 4;
    static const int NA = -1;
    array<int, Letters> next;
    bool patternEnd;

    Node()
    {
        fill(next.begin(), next.end(), NA);
        patternEnd = false;
    }
};

class Trie
{
private:
    Node root;
    int maxLength;

public:
    Trie()
    {
        maxLength = 0;
    }

    void insert(const string &pattern)
    {
        Node *currentNode = &root;
        for (char currentSymbol : pattern)
        {
            int index = letterToIndex(currentSymbol);

            if (currentNode->next[index] == Node::NA)
            {
                currentNode->next[index] = root.next.size();
                currentNode->next.fill(Node::NA);
            }

            currentNode = getNextNode(currentNode->next[index]);
        }

        currentNode->patternEnd = true;
        if (pattern.length() > maxLength)
        {
            maxLength = pattern.length();
        }
    }

    bool search(const string &text)
    {
        Node *currentNode = &root;
        for (char currentSymbol : text)
        {
            int index = letterToIndex(currentSymbol);
            if (currentNode->next[index] != Node::NA)
            {
                currentNode = getNextNode(currentNode->next[index]);
            }
            else
            {
                return false;
            }
        }
        return true;
    }

    int getMaxLength()
    {
        return maxLength;
    }

private:
    int letterToIndex(char letter)
    {
        switch (letter)
        {
        case 'A':
            return 0;
        case 'C':
            return 1;
        case 'G':
            return 2;
        case 'T':
            return 3;
        default:
            assert(false);
            return Node::NA;
        }
    }

    Node *getNextNode(int index)
    {
        if (index >= root.next.size())
        {
            return new Node();
        }
        else
        {
            return &root;
        }
    }
};

vector<int> solve(const string &text, int n, const vector<string> &patterns)
{
    vector<int> result;
    Trie trie;
    for (const string &pattern : patterns)
    {
        trie.insert(pattern);
    }

    for (int i = 0; i <= text.length() - trie.getMaxLength(); i++)
    {
        if (trie.search(text.substr(i)))
        {
            result.push_back(i);
        }
    }

    return result;
}

int main()
{
    string text;
    cin >> text;
    int n;
    cin >> n;
    vector<string> patterns(n);
    for (int i = 0; i < n; i++)
    {
        cin >> patterns[i];
    }

    vector<int> ans = solve(text, n, patterns);

    for (int j = 0; j < ans.size(); j++)
    {
        cout << ans[j];
        if (j + 1 < ans.size())
        {
            cout << " ";
        }
        else
        {
            cout << endl;
        }
    }

    return 0;
}
