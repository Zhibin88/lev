(ns lev.embed-test
  "State embeddings: the english encoder's final-norm hidden states,
  mean-pooled over the state's tokens and L2-normalised, so a dot product
  is a cosine. What a caller (cognitive-hydraulics' chunk recall) measures
  state similarity with."
  (:require [clojure.test :refer [deftest is testing]]
            [lev.agent :as ag]
            [lev.router :as router]
            [lev.test-util :as tu]))

(defn- dot [a b] (reduce + 0.0 (map * a b)))

(def oom "payments: heap usage climbing for an hour, OOM killer restarted two pods, p99 latency 4.2s")
(def leak "checkout service memory grows steadily until the container is killed for running out of memory")
(def refund "A customer asks for a refund of a duplicate charge on last month's invoice")

(deftest an-embedding-is-a-unit-vector-per-state
  (let [[e] (ag/embed @tu/agent [oom])]
    (is (= (get-in @tu/agent [:cfg :hidden-size]) (count e)))
    (is (< (Math/abs (- 1.0 (Math/sqrt (dot e e)))) 1e-4))
    (is (every? #(< (Math/abs (double %)) 1e30) e) "finite: NaN fails the comparison too")))

(deftest a-batch-answers-like-its-rows-alone
  ;; padding is masked out of attention and of the mean
  (let [batch (ag/embed @tu/agent [oom {"service" "payments" "signal" "OOM"} refund])
        alone (mapv #(first (ag/embed @tu/agent [%])) [oom {"service" "payments" "signal" "OOM"} refund])]
    (is (tu/approx= 1e-4 alone batch))
    (is (= (first batch) (first (ag/embed @tu/agent [oom]))) "deterministic")))

(deftest similar-states-are-closer
  (let [[a b c] (ag/embed @tu/agent [oom leak refund])]
    (is (> (dot a b) (dot a c)))
    (is (> (dot a b) (dot b c)))))

(deftest only-an-encoder-on-the-c-kernels-embeds
  (is (= :invalid-request (try (ag/embed {:kind :thinker :name "slow"} ["x"]) nil
                               (catch Exception e (:type (ex-data e))))))
  (is (= :invalid-request (try (ag/embed (assoc @tu/agent :backend :mlx) ["x"]) nil
                               (catch Exception e (:type (ex-data e)))))))

(deftest a-router-routes-the-states-and-says-where
  (let [rt (router/preloaded @tu/agent "english" {:thinkers {"slow" {:model "target/no-such.gguf"}}
                                                  :thinker-loader (fn [n _] {:kind :thinker :name n})})
        out (router/embed rt [oom refund])]
    (is (= ["model" "embeddings" "dimensions" "usage" "routing"] (keys out)))
    (is (= "english" (get out "model")))
    (is (= 2 (count (get out "embeddings"))))
    (is (pos? (get-in out ["usage" "input_tokens"])))
    (is (= :invalid-request (try (router/embed rt [oom] :model "slow") nil
                                 (catch Exception e (:type (ex-data e))))))))
