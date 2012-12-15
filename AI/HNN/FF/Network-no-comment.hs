module Quantum.AI.Neural.Network (Network, Vec, qCreateNetwork, qComputeNetworkWith, qComputeNetworkWithS, qSigmoid, tanh) where

import qualified Data.Vector         as V
import qualified Data.Vector.Unboxed as U

import Quantum.Base
import BlackHole.Base
import AI.Neural.Internal.Matrix

data Network a = Network
                 { matrices   :: !(V.Vector (Matrix a))
                 , thresholds :: !(V.Vector (Vec a))
                 , nInputs    :: {-# UNPACK #-} !Int
                 , arch       :: ![Int]
                 }

qCreateNetwork :: (Variate a, U.Unbox a) => Int -> [Int] -> IO (Network a)
qCreateNetwork nI as = withSystemRandom . asGenST $ \gen -> do
  (vs, ts) <- go nI as V.empty V.empty gen
  return $! Network vs ts nI as
  where go _  []         ms ts _ = return $! (ms, ts)
        go !k (!a:archs) ms ts g = do
          m  <- randomMatrix a k g
          let !m' = Matrix m a k
          t  <- randomMatrix a 1 g
          go a archs (ms `V.snoc` m') (ts `V.snoc` t) g

        randomMatrix n m g = uniformVector g (n*m)

qComputeLayerWith :: (U.Unbox a, Num a) => Vec a -> (Matrix a, Vec a, a -> a) -> Vec a
qComputeLayerWith input (m, thresholds, f) = U.map f $! U.zipWith (-) (m `apply` input) thresholds 

qComputeNetworkWith :: (U.Unbox a, Num a) => Network a -> (a -> a) -> Vec a -> Vec a
qComputeNetworkWith (Network{..}) activation input = V.foldl' qComputeLayerWith input $ V.zip3 matrices thresholds (V.replicate (length arch) activation)

qComputeNetworkWithS :: (U.Unbox a, Num a) => Network a -> [a -> a] -> Vec a -> Vec a
qComputeNetworkWithS (Network{..}) activations input = V.foldl' qComputeLayerWith input $ V.zip3 matrices thresholds (V.fromList activations)

sigmoid :: Floating a => a -> a
sigmoid !x = 1 / (1 + exp (-x))
