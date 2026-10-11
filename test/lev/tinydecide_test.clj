(ns lev.tinydecide-test
  "lev.tinydecide against TinyDecide's own conformance suite (its
  JavaScript reference engine's outputs, bench/tinydecide_golden.py ->
  golden/tinydecide/). Needs the release (meta.json + model.bin) under
  LEV_TEST_TINYDECIDE, else ~/src/models/TinyDecide; skipped without it."
  (:require [clojure.edn :as edn]
            [clojure.java.io :as io]
            [clojure.test :refer [deftest is testing]]
            [lev.agent :as ag]
            [lev.router :as router]
            [lev.sequence :as seq]
            [lev.think :as think]
            [lev.test-util :as tu]
            [lev.tinydecide :as td]))

(def td-dir
  (let [home (System/getProperty "user.home")]
    (some #(when (and % (.exists (io/file % "model.bin"))) %)
          [(System/getenv "LEV_TEST_TINYDECIDE") (str home "/src/models/TinyDecide")])))

(def model (delay (td/tinydecide {:name "tinydecide" :model td-dir})))

(def release (delay (:m @model)))

(defn- golden [name] (edn/read-string (slurp (str tu/golden-dir "/tinydecide/" name))))

(deftest tokenizes-as-the-reference
  (when td-dir
    (let [bad (remove (fn [{:keys [text ids]}] (= ids (td/tokenize @release text))) (golden "tok.edn"))]
      (is (empty? bad) (pr-str (map (fn [{:keys [text ids]}] {:text text :want ids :got (td/tokenize @release text)})
                                    (take 3 bad)))))))

(defn- matches-the-reference [release requests]
  (doseq [{:keys [name state questions ids truncated answers]} requests]
    (let [enc (td/encode-request release state questions)
          got (td/answer release state questions)]
      (is (= truncated (:truncated enc)) name)
      (when ids (is (= ids (:ids enc)) name))
      (is (= (mapv #(select-keys % [:type :pick]) answers) (mapv #(select-keys % [:type :pick]) got)) name)
      (is (tu/approx= 1e-4 answers (mapv #(select-keys % (keys %2)) got answers))
          (pr-str {:name name :want answers :got got})))))

(deftest answers-as-the-reference
  (when td-dir
    (matches-the-reference @release (golden "requests.edn"))))

(deftest the-full-build-answers-as-the-reference
  (when (and td-dir (td/release? (str td-dir "/full")))
    (let [full (td/load-release (str td-dir "/full"))]
      (try (matches-the-reference full (golden "requests.full.edn"))
           (finally (td/free-release! full))))))

(deftest refuses-what-the-reference-refuses
  (when td-dir
    (doseq [{:keys [name state questions]} (golden "errors.edn")]
      (let [e (try (td/answer @release state questions) nil (catch Exception e e))]
        (is (= :invalid-question (:type (ex-data e))) name)))))

(def readme-questions
  (array-map "app" {"type" "choice" "instructions" "Which app should handle this?"
                    "criteria" ["reminders" "music" "calendar" "restaurants" "weather"]}
             "urgent" {"type" "noul" "instructions" "The message is urgent."}
             "tone" {"type" "score" "instructions" "How positive is the tone?"
                     "criteria" ["negative" "neutral" "positive"]}))

(def readme-state "Book a table for 4 at an Italian place near the station on Friday at 7:30")

(deftest answers-a-request-in-the-jev-shape
  (when td-dir
    (let [out (ag/system-one @model readme-state readme-questions nil)
          a (get out "answers")
          want (:answers (first (golden "requests.edn")))]
      (is (= "tinydecide" (get out "model")))
      (is (= ["app" "urgent" "tone"] (vec (keys a))))
      (is (= "restaurants" (get-in a ["app" "choice"])))
      (testing "the reference's probabilities, rounded as every answer is"
        (is (tu/approx= 1e-4 (:probs (nth want 0)) (vec (vals (get-in a ["app" "probabilities"])))))
        (is (tu/approx= 1e-4 (:p (nth want 1)) (get-in a ["urgent" "noul"])))
        (is (tu/approx= 1e-4 (* 2 (:score (nth want 2))) (get-in a ["tone" "score"])) "lev's score is the level index"))
      (is (= 61 (get-in out ["usage" "input_tokens"])) "the readme request less its span question"))))

(deftest a-long-state-reports-truncation
  (when td-dir
    (let [out (ag/system-one @model (apply str (repeat 80 "the order arrived late again. "))
                             (select-keys readme-questions ["urgent"]) nil)
          dropped (get-in out ["truncated" "urgent"])]
      (is (pos-int? dropped) (pr-str out)))))

(deftest constraints-decide-jointly
  (when td-dir
    (let [out (ag/system-one @model readme-state readme-questions
                             {:constraints [["implies" ["app" "restaurants"] ["urgent" true]]]})]
      (is (contains? out "constraints"))
      (is (contains? (get-in out ["answers" "app"]) "decided")))))

(deftest too-many-options-is-a-question-error
  (when td-dir
    (let [e (try (ag/system-one @model "x" {"q" {"type" "choice" "instructions" "pick"
                                                 "criteria" (mapv str (range 33))}} nil)
                 nil (catch Exception e e))]
      (is (= :invalid-question (:type (ex-data e))))
      (is (= "q" (:qid (ex-data e)))))))

(deftest a-thinker-config-names-the-tinydecide-engine
  (when td-dir
    (let [a (think/thinker {:engine "tinydecide" :model td-dir :name "tiny"})]
      (is (= :tinydecide (:kind a)))
      (is (= "tiny" (:name a)))))
  (testing "a directory without the release is model-unavailable, before anything loads"
    (let [e (try (think/thinker {:engine "tinydecide" :model "target/nope"}) nil (catch Exception e e))]
      (is (= :model-unavailable (:type (ex-data e))))))
  (testing "the router serves it without the llm native, the release directory is what it checks"
    (let [r (router/make-router {:thinkers {"tiny" {:engine "tinydecide" :model (or td-dir "target/nope")}}})]
      (is (= (boolean td-dir) (router/available? r "tiny"))))))

(deftest a-structured-state-is-read-as-json
  (when td-dir
    (let [state (array-map "order" (array-map "status" "late" "days" 3))]
      (is (= (get (ag/system-one @model (seq/json-str state) (select-keys readme-questions ["urgent"]) nil) "answers")
             (get (ag/system-one @model state (select-keys readme-questions ["urgent"]) nil) "answers"))))))

(deftest a-choice-reads-its-descriptions
  (when td-dir
    (let [described {"app" {"type" "choice" "instructions" "Which app should handle this?"
                            "criteria" (array-map "a" "reminders" "b" "music" "c" "calendar" "d" "restaurants" "e" "weather")}}
          out (ag/system-one @model readme-state described nil)
          want (:probs (first (:answers (first (golden "requests.edn")))))]
      (is (= "d" (get-in out ["answers" "app" "choice"])))
      (is (tu/approx= 1e-4 want (vec (vals (get-in out ["answers" "app" "probabilities"]))))))))
