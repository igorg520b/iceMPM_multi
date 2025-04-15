#ifndef SIMPLEOBJECTPOOL_H
#define SIMPLEOBJECTPOOL_H

#include <vector>
#include <queue>

template<class T>
class SimplePool
{
public:
    SimplePool(int initialSize);
    ~SimplePool();
    SimplePool& operator=(SimplePool&) = delete;

    T* take();
    void release(T* p) {available.push(p);}
    void release(std::vector<T*> &vec);
    void releaseAll();
    void printout(); // for testing

private:
    std::queue<T*> available;      // items that are free to use
    std::vector<T*> registry;       // all items of the pool
};

template<class T>
SimplePool<T>::SimplePool(int initialSize)
{
    registry.reserve(initialSize*2);
    for(int i=0;i<initialSize;i++)
    {
        T* obj = new T;
        available.push(obj);
        registry.push_back(obj);
    }
}

template<class T>
SimplePool<T>::~SimplePool()
{
    for(auto &x : registry) delete x;
}

template <class T>
T* SimplePool<T>::take()
{
    T *p;
    if(!available.empty()) { p = available.front(); available.pop(); }
    else
    {
        p = new T;
        registry.push_back(p);
    }
    return p;
}

template<class T>
void SimplePool<T>::releaseAll()
{
    available.clear();
    for(unsigned i=0;i<registry.size();i++) available.push(registry[i]);
}

template<class T>
void SimplePool<T>::release(std::vector<T*> &vec)
{
    for(T* p : vec) available.push(p);
    vec.clear();
}

#endif
