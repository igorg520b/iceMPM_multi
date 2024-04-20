#ifndef KDOP24_H
#define KDOP24_H


struct kDOP24
{
    /*
      k=18 (Aabb + 12 diagonal planes that "cut off" some space of the edges):
  (-1,0,0) and (1,0,0)  -> indices 0 and 9
  (0,-1,0) and (0,1,0)  -> indices 1 and 10
  (0,0,-1) and (0,0,1)  -> indices 2 and 11
  (-1,-1,0) and (1,1,0) -> indices 3 and 12
  (-1,0,-1) and (1,0,1) -> indices 4 and 13
  (0,-1,-1) and (0,1,1) -> indices 5 and 14
  (-1,1,0) and (1,-1,0) -> indices 6 and 15
  (-1,0,1) and (1,0,-1) -> indices 7 and 16
  (0,-1,1) and (0,1,-1) -> indices 8 and 17
*/
    float d[24];
    float ctrX, ctrY, ctrZ;
    float dX, dY, dZ;

    void Reset();
    bool Overlaps(kDOP24 &b);
    void Expand(float x, float y, float z);
    void Expand(kDOP24 &b);

    inline void MinMax(float p, float &mi, float &ma);
};

#endif // KDOP24_H
