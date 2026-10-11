;; Which pooling of the english encoder makes a usable state embedding
;; (lev.agent/embed): 12 short incident / billing states in 4 groups,
;; three phrasings each. Mean, CLS and mean-without-specials, raw and
;; centered on the set's own mean; "nn" is how many states have a
;; nearest neighbour from their own group. bench/README.md, embeddings.
;;
;;   jolt bench/embeddings.clj

(require '[lev.agent :as ag] '[lev.model :as m] '[lev.tensors :as t] '[lev.sequence :as seq] '[jolt.ffi :as ffi])
(def a (ag/load-agent "data"))
(defn dot [x y] (reduce + 0.0 (map * x y)))
(defn unit [v] (let [n (Math/sqrt (dot v v))] (mapv #(/ % n) v)))
;; groups: same cause in different words should cluster
(def groups
  {:leak ["heap climbing for 2h, OOM kills, latency p99 4.2s"
          "memory grows steadily until the container is killed for running out of memory"
          "RSS keeps rising after each request; pods restarted by the OOM killer"]
   :config ["error rate 9% since the deploy 14 min ago that touched config/limits.toml"
            "errors started right after a config change shipped; bad value in the limits file"
            "a deploy changed the connection limits config and requests began failing"]
   :capacity ["CPU saturated on every replica, request queue depth 900"
              "traffic doubled, all instances at 100% CPU, requests queueing"
              "load spike: thread pools exhausted across the fleet, latency rising with traffic"]
   :billing ["customer asks for a refund of a duplicate charge"
             "invoice shows the wrong amount, please correct the bill"
             "I was charged twice for my subscription this month"]})
(def items (for [[g xs] groups x xs] [g x]))
(defn pools [state]
  (let [{:keys [cfg tok w]} a sp (:specials tok)
        ids (into [(:cls sp)] (conj (vec (seq/encode-state tok state)) (:sep sp)))
        n (count ids)]
    (with-open [ar (ffi/confined-arena)]
      (binding [t/*arena* ar]
        (let [ws (m/workspace cfg 1 n)
              att (t/from-bytes (repeat n 1) [1 n])
              h (m/encode-batch w cfg (t/from-ints ids) att (m/row-masks cfg att 1 n) ws)]
          {:mean (t/to-floats (t/mean-rows (t/rows h 0 n)))
           :cls (t/to-floats (t/rows h 0 1))
           :inner (t/to-floats (t/mean-rows (t/rows h 1 (- n 2))))})))))
(def ps (mapv (fn [[g x]] [g (pools x)]) items))
(defn score [k center?]
  (let [vs (mapv #(get (second %) k) ps)
        mu (mapv #(/ % (count vs)) (apply map + vs))
        vs (mapv #(unit (if center? (mapv - % mu) %)) vs)
        gs (mapv first ps)
        pairs (for [i (range (count vs)) j (range (count vs)) :when (< i j)] [(= (gs i) (gs j)) (dot (vs i) (vs j))])
        same (map second (filter first pairs)) diff (map second (remove first pairs))
        avg #(/ (reduce + %) (count %))
        ;; nearest neighbour accuracy: is each item's nearest other item in its group?
        nn (count (filter identity (for [i (range (count vs))]
                                     (let [j (apply max-key #(if (= % i) -9 (dot (vs i) (vs %))) (range (count vs)))]
                                       (= (gs i) (gs j))))))]
    (println (name k) (if center? "centered" "raw     ") "same" (format "%.3f" (avg same)) "diff" (format "%.3f" (avg diff))
             "nn" nn "/" (count vs))))
(doseq [k [:mean :cls :inner] c [false true]] (score k c))
