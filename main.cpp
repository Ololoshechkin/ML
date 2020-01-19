#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <tuple>
#include <fstream>
#include <random>
#include <memory>

int NONE = -1;
using Features = std::vector<double>;
using FeaturesList = std::vector<Features>;
using Classes = std::vector<int>;
using Frequences = Classes;
using Object = std::pair<Features, int>;
using Objects = std::vector<Object>;

Objects zip(FeaturesList fs, Classes cs) {
  Objects objs;
  objs.reserve(fs.size());
  for (size_t i = 0; i < fs.size(); ++i) {
    objs.emplace_back(fs[i], cs[i]);
  }
  return objs;
}

class Node {
 public:
  std::unique_ptr<Node> left = nullptr;
  std::unique_ptr<Node> right = nullptr;
  size_t index = static_cast<size_t>(NONE);
  double threshold = NONE;
  int result = NONE;
  FeaturesList features;
  Classes classes;
  
  explicit Node(
          int result = NONE, FeaturesList features = {}, Classes classes = {},
          size_t index = static_cast<size_t>(NONE), double threshold = NONE
  ) :
          left(nullptr), right(nullptr), result(result), features(features),
          classes(classes), index(index), threshold(threshold) {}
  
  int classify(const Features& features) const {
    if (result != NONE)
      return result;
    return features[index] < threshold ? left->classify(features)
                                       : right->classify(features);
  }
  
  size_t size() const {
    return result != NONE ? 1 : 1 + left->size() + right->size();
  }
  
  int write(int v = 1, bool prnt = true) const {
    if (result != NONE) {
      if (prnt)
        std::cout << "C " << result << '\n';
      return v;
    }
    int l = v + 1;
    int mxl = left->write(l, false);
    int r = mxl + 1;
    if (prnt)
      std::cout << "Q " << index + 1 << " " << threshold << " " << l << " " << r
                << '\n';
    left->write(l, prnt);
    int mxr = right->write(r, prnt);
    return mxr;
  }
  
  ~Node() {
  }
};

using Tree = std::unique_ptr<Node>;
using Forest = std::vector<Tree>;

Frequences bincount(const Classes& classes) {
  if (classes.empty())
    return {};
  int sz = *std::max_element(classes.begin(), classes.end()) + 1;
  Frequences frequences(static_cast<unsigned long>(sz));
  for (auto c : classes) {
    ++frequences[c];
  }
  return frequences;
}

template<typename List>
double sum(List l) {
  return std::accumulate(l.begin(), l.end(), 0.0);
}

template<typename List, typename Transform>
double sum(List l, Transform f) {
  return std::accumulate(
          l.begin(), l.end(), 0.0, [&f](double res, double cur) {
            return res + f(cur);
          });
}

template<typename List>
decltype(auto) argmax(List x) {
  return std::distance(x.begin(), std::max_element(x.begin(), x.end()));
}

double giniGain(const Frequences& frequences) {
  double s = sum(frequences);
  if (s == 0.0)
    return 1.0;
  double sumP = sum(frequences, [](double f) { return f * f; }) / (s * s);
  return 1.0 - sumP;
}

double informativeness(
        const Frequences& frequences, const Frequences& frequences1,
        const Frequences& frequences2
) {
  double sz1 = sum(frequences1);
  double sz2 = sum(frequences2);
  double sz = sz1 + sz2;
  return giniGain(frequences) - giniGain(frequences1) * sz1 / sz -
         giniGain(frequences2) * sz2 / sz;
}

std::pair<size_t, double>
split(FeaturesList& features, Classes& classes, bool bagging = false) {
  size_t n = features[0].size();
  size_t m = features.size();
  double bestInf = -1e10;
  size_t bestIndex = 0;
  double bestTreshold = 0.0;
  auto sqN = static_cast<size_t>(sqrt(n));
  for (size_t j = 0; j < n; ++j) {
    if (bagging && rand() % n > sqN)
      continue;
    auto objects = zip(features, classes);
    std::sort(
            objects.begin(), objects.end(),
            [j](const Object& lhs, const Object& rhs) -> bool {
              return lhs.first[j] < rhs.first[j];
            });
    for (size_t i = 0; i < objects.size(); ++i) {
      features[i] = objects[i].first;
      classes[i] = objects[i].second;
    }
    for (size_t i = 0; i < m - 1; ++i) {
      double treshold = (features[i][j] + features[i + 1][j]) / 2.0;
      auto mid = i + (features[i][j] < treshold);
      Classes classes1(classes.begin(), classes.begin() + mid);
      Classes classes2(classes.begin() + mid, classes.end());
      double inf = informativeness(
              bincount(classes), bincount(classes1), bincount(classes2));
      if (inf > bestInf) {
        bestInf = inf;
        bestIndex = j;
        bestTreshold = treshold;
      }
    }
  }
  return std::make_pair(bestIndex, bestTreshold);
}

Classes nonzero(const Frequences& frequences) {
  Classes nz;
  for (int i = 0; i < int(frequences.size()); ++i) {
    if (frequences[i] != 0)
      nz.push_back(i);
  }
  return nz;
}

Tree buildTree(
        FeaturesList& features, Classes& classes, int maxLength,
        bool bagging = false
) {
  if (features.empty()) {
    std::cout << "EMPTY!!!" << std::endl;
    return std::make_unique<Node>(0);
  }
  auto classFrequencies = bincount(classes);
  if (maxLength == 0) {
    int maxClass = static_cast<int>(argmax(classFrequencies));
    return std::make_unique<Node>(maxClass, features, classes);
  }
  auto existingClasses = nonzero(classFrequencies);
  if (existingClasses.size() == 1)
    return std::make_unique<Node>(existingClasses[0], features, classes);
  auto tmp = split(features, classes, bagging);
  size_t index = tmp.first;
  double threshold = tmp.second;
  auto cur = std::make_unique<Node>(NONE, features, classes, index, threshold);
  FeaturesList features1, features2;
  Classes classes1, classes2;
  for (size_t i = 0; i < features.size(); ++i) {
    if (features[i][index] < threshold) {
      features1.push_back(features[i]);
      classes1.push_back(classes[i]);
    } else {
      features2.push_back(features[i]);
      classes2.push_back(classes[i]);
    }
  }
  cur->left = buildTree(features1, classes1, maxLength - 1);
  cur->right = buildTree(features2, classes2, maxLength - 1);
  return cur;
}

void upgradeTree(
        Tree& tree, int maxLength, bool bagging = false
) {
  if (tree->result != NONE && maxLength != 0) {
    tree = std::move(
            buildTree(tree->features, tree->classes, maxLength, bagging));
    return;
  }
  upgradeTree(tree->left, maxLength - 1);
  upgradeTree(tree->right, maxLength - 1);
}

decltype(auto) readDataset() {
  int m, k, h;
  std::cin >> m >> k >> h;
  size_t n;
  std::cin >> n;
  FeaturesList features;
  Classes classes;
  for (size_t i = 0; i < n; ++i) {
    Features fs(static_cast<unsigned long>(m));
    for (size_t j = 0; j < m; ++j) {
      std::cin >> fs[j];
    }
    int c;
    std::cin >> c;
    features.push_back(fs);
    classes.push_back(c);
  }
  return std::make_tuple(features, classes, h);
}

decltype(auto) readDatasetLab(const std::string& filename) {
  std::ifstream fin;
  fin.open(filename);
  int m, k;
  fin >> m >> k;
  size_t n;
  fin >> n;
  FeaturesList features;
  Classes classes;
  for (size_t i = 0; i < n; ++i) {
    Features fs(static_cast<unsigned long>(m));
    for (size_t j = 0; j < m; ++j) {
      fin >> fs[j];
    }
    int c;
    fin >> c;
    features.push_back(fs);
    classes.push_back(c);
  }
  fin.close();
  return std::make_tuple(features, classes);
}

template<typename F>
double qualityImpl(
        const FeaturesList& features, const Classes& classes, F isCorrectClass
) {
  double diff = 0.0;
  for (size_t i = 0; i < features.size(); ++i) {
    if (!isCorrectClass(features[i], classes[i]))
      diff += 1.0;
  }
  return 1.0 - diff / double(features.size());
}

Forest
buildForest(FeaturesList& features, Classes& classes, int length, size_t size) {
  Forest trees;
  trees.reserve(size);
  size_t n = features.size();
  for (size_t i = 0; i < size; ++i) {
    FeaturesList subFeatures;
    subFeatures.reserve(n);
    Classes subClasses;
    subClasses.reserve(n);
    for (size_t j = 0; j < n; ++j) {
      size_t id = static_cast<size_t>(rand() % n);
      subFeatures.push_back(features[id]);
      subClasses.push_back(classes[id]);
    }
    trees.emplace_back(buildTree(subFeatures, subClasses, length, true));
  }
  return trees;
}

void upgradeForest(
        Forest& trees, int length
) {
  std::for_each(
          trees.begin(), trees.end(), [length](auto& tree) {
            upgradeTree(tree, length, true);
          });
}

int classify(const Tree& tree, const Features& features) {
  return tree->classify(features);
}

int classify(const Forest& forest, const Features& features) {
  Classes classes;
  classes.reserve(forest.size());
  for (const auto& tree : forest) {
    classes.push_back(tree->classify(features));
  }
  auto frequences = bincount(classes);
  return static_cast<int>(argmax(frequences));
}

template<typename Classifier>
double quality(
        const Classifier& classifier, const FeaturesList& features,
        const Classes& classes
) {
  return qualityImpl(
          features, classes, [&classifier](const auto& f, int c) {
            return classify(classifier, f) == c;
          });
}

int main() {
  srand(time(0));
  std::istream::sync_with_stdio(false);
  std::cin.tie(nullptr);
  for (int i = 1; i <= 21; ++i) {
    char buffer1[100];
    sprintf(buffer1, "/Users/Vadim/Documents/ML/DT_txt/%02d_train.txt", i);
    char buffer2[100];
    sprintf(buffer2, "/Users/Vadim/Documents/ML/DT_txt/%02d_test.txt", i);
    std::vector<int> ls;
    std::vector<double> qs;
    std::vector<double> qsF;
    //
    auto tmpTrain = readDatasetLab(std::string(buffer1));
    auto featuresTrain = std::get<FeaturesList>(tmpTrain);
    auto classesTrain = std::get<Classes>(tmpTrain);
    //
    auto tmpTest = readDatasetLab(std::string(buffer2));
    auto featuresTest = std::get<FeaturesList>(tmpTest);
    auto classesTest = std::get<Classes>(tmpTest);
    //
    auto tree = buildTree(featuresTrain, classesTrain, 1);
    //
    auto forest = buildForest(featuresTrain, classesTrain, 1, 10);
    //
    for (int length = 1;
         length <= 10; ++length, upgradeTree(tree, length), upgradeForest(
            forest, length)) {
      ls.push_back(length);
      qs.push_back(quality(tree, featuresTest, classesTest));
      qsF.push_back(quality(forest, featuresTest, classesTest));
    }
    //
    std::cout << "import numpy as np\nbestH = []\n\nls = [";
    for (int l : ls) {
      std::cout << l << ", ";
    }
    std::cout << "]\nqs = [";
    for (double q : qs) {
      std::cout << q << ", ";
    }
    std::cout << "]\nqsF = [";
    for (double q : qsF) {
      std::cout << q << ", ";
    }
    std::cout << "]\nprint(\'test = {0}\'.format(" << i
              << "))\nplt.plot(ls, qs)\nplt.plot(ls, qsF)\nplt.show()\nbestH.append(np.argmax(qs) + 1)\n\n";
  }
  return 0;
}