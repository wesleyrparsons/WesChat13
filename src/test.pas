unit Test;

{$mode ObjFPC}{$H+}{$I proprietary.txt}

{ WesChat, Version 1.2, begun January 10, 2026, by Wesley R. Parsons, wespar@bellsouth.net, www.wesparsons.com.}

interface

uses
  Global,
  Matrix;

implementation

begin

  // SGEMM call, OpenBLAS 3.30.
  cblas_sgemm(
    101,               // Layout
    111,               // A not transposed
    111,               // B not transposed
    2,                 // M = rows of X
    1,                 // N = columns of Wq
    3,                 // K = shared dimension
    1.0,               // alpha
    @X[0],             // A
    3,                 // LDA = K for row-major
    @Wq[0],            // B
    1,                 // LDB = N for row-major
    0.0,               // beta
    @OutVec[0],        // C
    1
  );

end;

end.

