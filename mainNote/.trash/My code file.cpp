#include <bits/stdc++.h>
using namespace std;

void print_arr(long long a[], long long size);
long long max(long long a, long long b) {
    if (a > b) {
        return a;
    } else {
        return b;
    }
}

int main() {
    long long n; //
    cin >> n;    //
    long long pvr = n;
    long long temp = 1;                  //
    long long arr[42844285];                //
    for (long long i = 0; i <= n; i++) { //
        arr[i] = 0;                      //
    } //
    cout << "WAWASWA";
    for (long long i = 2; i <= pvr; i++) {
        while (n % i == 0) {
            if (n < i || n == 1)
                break;
            n /= i;
            arr[i]++;
            temp = max(temp, i);
        }
    }
    print_arr(arr, pvr + 1);
    for (long long i = 2; i <= pvr; i++) {
        if (arr[i] != 0 && arr[i] != 1) {
            cout << i << '^' << arr[i];
            if (i != temp) {
                cout << " * ";
            }
        } else if (arr[i] == 1) {
            cout << i;
            if (i != temp) {
                cout << " * ";
            }
        }
    }
    // cout << '\n';
    // cout << temp;
    // for (long long i = 0; i <= pvr; i++) {
    //     cout << arr[i] << endl;
    // }
    return 0;
}

void print_arr(long long a[], long long size) {
    for (int i = 0; i < size; i++) {
        cout << a[i] << endl;
    }
}
