# Problem 1

We saw how to solve TSPs in this module and in particular presented two approaches to encode a TSP as an integer linear program. In this problem, we will ask you to adapt the TSP solution to the related problem of $k$ Travelling Salespeople Problem ($k$-TSP).

Let $G$ be a complete graph with $n$ vertices that we will label $0, \ldots, n-1$ (keeping Python array indexing in mind). Our costs are specified using a matrix $C$ wherein $C_{i,j}$ is the cost of the edge from vertex $i$ to $j$ for $i \not= j$.

In this problem, we have $k \geq 1$ salespeople who must  start from vertex $0$ of the graph (presumably the location of the sales office) and together visit every location in the graph,  each returning back to vertex $0$. Each location must be visited exactly once by some salesperson in the team. Therefore, other than vertex $0$ (the start/end vertex of every salesperson's tour), no two salesperson tours have a vertex in common. Notice that for $k=1$, this is just the regular TSP problem we have studied. 

Also, all $k$ salespeople must be employed in the tour. In other words, if we have $k=3$ then each salesperson must start at $0$ and visit a sequence of one or more vertices and come back to $0$. No salesperson can be "idle".

## Example-1

Consider a graph with $5$ nodes and the following cost matrix:

$$ \begin{array}{c|ccccc}
  \text{Vertices} & 0 & 1 & 2 & 3 & 4 \\ 
   \hline
 0 & - & 3 & 4 & 3 & 5 \\ 
 1 & 1 & - & 2 & 4 & 1 \\ 
 2 & 2 & 1 & - & 5 & 4 \\ 
 3 & 1 & 1 & 5 & - & 4 \\ 
 4 & 2 & 1 & 3 & 5 & - \\ 
 \end{array}$$
 
 For instance $C_{2,3}$ the cost of edge from vertex $2$ to $3$ is $5$. The $-$ in the diagonal entries simply tells us that we do not care what goes in there since we do not have self-loops in the graph.
 
The optimal $2$-TSP tour for $k=2$ salespeople is shown below.
  - Salesperson # 1: $0 \rightarrow 2 \rightarrow 1 \rightarrow 4 \rightarrow 0$.
  - Salesperson # 2: $0 \rightarrow 3 \rightarrow 0$.
  
The total cost of the edges traversed by the two salespeople equals $12$.

For $k=3$, the optimal $3-$ TSP tour is as shown below.
  - Salesperson # 1: $0 \rightarrow 1 \rightarrow 4$, 
  - Salesperson # 2: $0 \rightarrow 2$, 
  - Salesperson # 3: $0 \rightarrow 3$.

The total cost is $16$.

The objective of this problem is to formulate an ILP using the MTZ approach.

### Problem 1A (MTZ approach)

We will use the same ILP setup as in our notes (see the notes on Exact Approaches to TSP that includes the ILP encodings we will use in this problem).
  - Decision variables $x_{i,j}$ for $i \not= j$ denoting that the tour traverses the edge from $i$ to $j$.
  - Time stamps $t_1, \ldots, t_{n-1}$. The start/end vertex $0$ does not get a time stamp.
  
Modify the MTZ approach to incorporate the fact that $k$ salespeople are going to traverse the graph.

#### (A) Degree Constraints

What do the new degree constraints look like? Think about how many edges in the tour will need to enter/leave each vertex? Note that you may have to treat vertex $0$ differently from the other vertices of the graph.

Your answer below is not graded. However you are encouraged to write it down and check with the answers to select problems provided at the end.

YOUR ANSWER HERE

### (B) Time Stamp Constraints 

Formulate the time stamp constraints for the $k$-TSP problem. Think about how you would need to change them to eliminate subtour.

Your answer below is not graded. However you are encouraged to write it down and check with the answers to select problems provided at the end.


YOUR ANSWER HERE

### (C) Implement

Complete the implementation of the function `k_tsp_mtz_encoding(n, k, cost_matrix)` below. It follows the same input convention as the code supplied in the notes. The input `n` denotes the size of the graph with vertices labeled `0`,.., `n-1`, `k` is the number of salespeople, and `cost_matrix` is a list of lists wherein `cost_matrix[i][j]` is the edge cost to go from `i` to `j` for `i != j`. Your code must avoid accessing `cost_matrix[i][i]` to avoid bugs. These entries will be supplied as `None` in the test cases.

Your code must return a list `lst` that has exactly $k$ lists in it, wherein `lst[j]` represents the locations visited by the $j^{th}$ salesperson. 

For the example above, for $k=2$, your code must return 
~~~
[ [0, 2, 1, 4], [0, 3] ]
~~~
For the example above, for $k=3$, your code must return
~~~
[ [0, 1, 4], [0, 2], [0, 3] ]
~~~


```python
from pulp import *

def k_tsp_mtz_encoding(n, k, cost_matrix):
    # check inputs are OK
    assert 1 <= k < n
    assert len(cost_matrix) == n, f'Cost matrix is not {n}x{n}'
    assert all(len(cj) == n for cj in cost_matrix), f'Cost matrix is not {n}x{n}'

    prob = LpProblem('kTSP', LpMinimize)

    # Decision variables
    x = {(i, j): LpVariable(f'x_{i}_{j}', cat='Binary') for i in range(n) for j in range(n) if i != j}
    t = {i: LpVariable(f't_{i}', lowBound=1, upBound=n, cat='Integer') for i in range(1, n)}

    # Objective function
    prob += sum(cost_matrix[i][j] * x[(i, j)] for i in range(n) for j in range(n) if i != j and cost_matrix[i][j] is not None)

    # Degree constraints
    for i in range(n):
        if i == 0:
            prob += sum(x[(0, j)] for j in range(1, n) if cost_matrix[0][j] is not None) == k
            prob += sum(x[(j, 0)] for j in range(1, n) if cost_matrix[j][0] is not None) == k
        else:
            prob += sum(x[(i, j)] for j in range(n) if j != i and cost_matrix[i][j] is not None) == 1
            prob += sum(x[(j, i)] for j in range(n) if j != i and cost_matrix[j][i] is not None) == 1

    # Time stamp constraints
    for i in range(1, n):
        for j in range(1, n):
            if i != j and cost_matrix[i][j] is not None:
                prob += t[i] - t[j] + (n * x[(i, j)]) <= n - 1

    # Solve the problem
    prob.solve()

    # Construct the solution
    solution = [[] for _ in range(k)]
    visited_nodes = set()
    current_salesperson = 0
    current_node = 0
    solution[current_salesperson].append(current_node)
    visited_nodes.add(current_node)

    while len(visited_nodes) < n:
        next_node = None
        min_cost = float('inf')
        for j in range(1, n):
            if cost_matrix[current_node][j] is not None and x[(current_node, j)].value() > 0.5 and j not in visited_nodes:
                if cost_matrix[current_node][j] < min_cost:
                    next_node = j
                    min_cost = cost_matrix[current_node][j]
        if next_node is not None:
            current_node = next_node
            solution[current_salesperson].append(current_node)
            visited_nodes.add(current_node)
        else:
            # If no more edges can be found, move to the next salesperson
            current_salesperson += 1
            if current_salesperson < k:
                current_node = 0
                solution[current_salesperson].append(current_node)
                visited_nodes.add(current_node)
            else:
                break

    # Calculate the total tour cost
    tour_cost = 0
    for tour in solution:
        if tour:
            i = 0
            for j in tour[1:]:
                tour_cost += cost_matrix[i][j]
                i = j
            tour_cost += cost_matrix[i][0]

    return solution
```


```python
cost_matrix=[ [None,3,4,3,5],
             [1, None, 2,4, 1],
             [2, 1, None, 5, 4],
             [1, 1, 5, None, 4],
             [2, 1, 3, 5, None] ]
n=5
k=2
all_tours = k_tsp_mtz_encoding(n, k, cost_matrix)
print(f'Your code returned tours: {all_tours}')
assert len(all_tours) == k, f'k={k} must yield two tours -- your code returns {len(all_tours)} tours instead'

tour_cost = 0
for tour in all_tours:
    assert tour[0] == 0, 'Each salesperson tour must start from vertex 0'
    i = 0
    for j in tour[1:]:
        tour_cost += cost_matrix[i][j]
        i = j
    tour_cost += cost_matrix[i][0]

print(f'Tour cost obtained by your code: {tour_cost}')
assert abs(tour_cost - 12) <= 0.001, f'Expected tour cost is 12, your code returned {tour_cost}'
for i in range(1, n):
    is_in_tour = [ 1 if i in tour else 0 for tour in all_tours]
    assert sum(is_in_tour) == 1, f' vertex {i} is in {sum(is_in_tour)} tours -- this is incorrect'

print('test passed: 3 points')

```

    Your code returned tours: [[0, 3], [0, 2, 1, 4]]
    Tour cost obtained by your code: 12
    test passed: 3 points



```python
cost_matrix=[ [None,3,4,3,5],
             [1, None, 2,4, 1],
             [2, 1, None, 5, 4],
             [1, 1, 5, None, 4],
             [2, 1, 3, 5, None] ]
n=5
k=3
all_tours = k_tsp_mtz_encoding(n, k, cost_matrix)
print(f'Your code returned tours: {all_tours}')
assert len(all_tours) == k, f'k={k} must yield two tours -- your code returns {len(all_tours)} tours instead'

tour_cost = 0
for tour in all_tours:
    assert tour[0] == 0, 'Each salesperson tour must start from vertex 0'
    i = 0
    for j in tour[1:]:
        tour_cost += cost_matrix[i][j]
        i = j
    tour_cost += cost_matrix[i][0]

print(f'Tour cost obtained by your code: {tour_cost}')
assert abs(tour_cost - 16) <= 0.001, f'Expected tour cost is 16, your code returned {tour_cost}'
for i in range(1, n):
    is_in_tour = [ 1 if i in tour else 0 for tour in all_tours]
    assert sum(is_in_tour) == 1, f' vertex {i} is in {sum(is_in_tour)} tours -- this is incorrect'

print('test passed: 2 points')
```

    Your code returned tours: [[0, 1, 4], [0, 3], [0, 2]]
    Tour cost obtained by your code: 16
    test passed: 2 points



```python
cost_matrix = [ 
 [None, 1, 1, 1, 1, 1, 1, 1],
    [0, None, 1, 2, 1, 1, 1, 1],
    [1, 0, None, 1, 2, 2, 2, 1],
    [1, 2, 2, None, 0, 1, 2, 1],
    [1, 1, 1, 1, None, 1, 1, 1],
    [0,  1, 2, 1, 1, None, 1, 1],
    [1, 0,  1, 2, 2, 2,None, 1],
    [1, 2, 2, 0, 1, 2, 1, None],
]
n = 8
k = 2

all_tours = k_tsp_mtz_encoding(n, k, cost_matrix)
print(f'Your code returned tours: {all_tours}')
assert len(all_tours) == k, f'k={k} must yield two tours -- your code returns {len(all_tours)} tours instead'

tour_cost = 0
for tour in all_tours:
    assert tour[0] == 0, 'Each salesperson tour must start from vertex 0'
    i = 0
    for j in tour[1:]:
        tour_cost += cost_matrix[i][j]
        i = j
    tour_cost += cost_matrix[i][0]

print(f'Tour cost obtained by your code: {tour_cost}')
assert abs(tour_cost - 4) <= 0.001, f'Expected tour cost is 4, your code returned {tour_cost}'
for i in range(1, n):
    is_in_tour = [ 1 if i in tour else 0 for tour in all_tours]
    assert sum(is_in_tour) == 1, f' vertex {i} is in {sum(is_in_tour)} tours -- this is incorrect'

print('test passed: 3 points')
```

    Your code returned tours: [[0, 6, 2, 1], [0, 7, 3, 4, 5]]
    Tour cost obtained by your code: 4
    test passed: 3 points



```python
cost_matrix = [ 
 [None, 1, 1, 1, 1, 1, 1, 1],
    [0, None, 1, 2, 1, 1, 1, 1],
    [1, 0, None, 1, 2, 2, 2, 1],
    [1, 2, 2, None, 0, 1, 2, 1],
    [1, 1, 1, 1, None, 1, 1, 1],
    [0,  1, 2, 1, 1, None, 1, 1],
    [1, 0,  1, 2, 2, 2,None, 1],
    [1, 2, 2, 0, 1, 2, 1, None],
]
n = 8
k = 4

all_tours = k_tsp_mtz_encoding(n, k, cost_matrix)
print(f'Your code returned tours: {all_tours}')
assert len(all_tours) == k, f'k={k} must yield two tours -- your code returns {len(all_tours)} tours instead'

tour_cost = 0
for tour in all_tours:
    assert tour[0] == 0, 'Each salesperson tour must start from vertex 0'
    i = 0
    for j in tour[1:]:
        tour_cost += cost_matrix[i][j]
        i = j
    tour_cost += cost_matrix[i][0]

print(f'Tour cost obtained by your code: {tour_cost}')
assert abs(tour_cost - 6) <= 0.001, f'Expected tour cost is 6, your code returned {tour_cost}'
for i in range(1, n):
    is_in_tour = [ 1 if i in tour else 0 for tour in all_tours]
    assert sum(is_in_tour) == 1, f' vertex {i} is in {sum(is_in_tour)} tours -- this is incorrect'

print('test passed: 2 points')
```

    Your code returned tours: [[0, 2], [0, 5], [0, 6, 1], [0, 7, 3, 4]]
    Tour cost obtained by your code: 6
    test passed: 2 points



```python
from random import uniform, randint

def create_cost(n):
    return [ [uniform(0, 5) if i != j else None for j in range(n)] for i in range(n)]

for trial in range(5):
    print(f'Trial # {trial}')
    n = randint(5, 11)
    k = randint(2, n//2)
    print(f' n= {n}, k={k}')
    cost_matrix = create_cost(n)
    print('cost_matrix = ')
    print(cost_matrix)
    all_tours = k_tsp_mtz_encoding(n, k, cost_matrix)
    print(f'Your code returned tours: {all_tours}')
    assert len(all_tours) == k, f'k={k} must yield two tours -- your code returns {len(all_tours)} tours instead'

    tour_cost = 0
    for tour in all_tours:
        assert tour[0] == 0, 'Each salesperson tour must start from vertex 0'
        i = 0
        for j in tour[1:]:
            tour_cost += cost_matrix[i][j]
            i = j
        tour_cost += cost_matrix[i][0]

    print(f'Tour cost obtained by your code: {tour_cost}')
    #assert abs(tour_cost - 6) <= 0.001, f'Expected tour cost is 6, your code returned {tour_cost}'
    for i in range(1, n):
        is_in_tour = [ 1 if i in tour else 0 for tour in all_tours]
        assert sum(is_in_tour) == 1, f' vertex {i} is in {sum(is_in_tour)} tours -- this is incorrect'
    print('------')
print('test passed: 15 points')
```

    Trial # 0
     n= 8, k=3
    cost_matrix = 
    [[None, 1.1523791757117547, 4.287776752343469, 1.0735578729423234, 2.6990609842239777, 3.182273033536553, 4.678502791664541, 1.1083271317595904], [0.35429775841147537, None, 0.3740783841891937, 0.011919209971588685, 3.4734171303754042, 4.04105397023974, 2.5240738699476055, 0.9701815299160649], [3.278687372697722, 1.4124541371017978, None, 2.2381738131516866, 0.6874041625395111, 1.1437390573426542, 3.7233164237996563, 4.498493001763831], [2.037227621183848, 2.8842281808664665, 2.353533106087559, None, 4.05782747796424, 4.430720213642045, 0.9115031184064148, 0.862798581845115], [4.241932906894838, 1.382124484093985, 4.250847869262001, 4.343861117684312, None, 3.120887234081373, 1.0912550595904291, 4.206420564728705], [3.0458559167254116, 2.07743023281307, 4.552099987522319, 2.831645926493545, 2.857821003586837, None, 1.361561164367071, 0.02422932863415883], [3.3628949362557927, 4.643969809377406, 2.5410311292151104, 4.741733466918544, 1.6243031252522817, 3.5687556272371292, None, 4.008070577531929], [4.844242419201607, 4.47362707025861, 4.075005566096911, 1.170848847437902, 1.5950936330755294, 1.1493367326686537, 3.805087676180607, None]]
    Your code returned tours: [[0, 3], [0, 7, 5], [0, 1, 2, 4, 6]]
    Tour cost obtained by your code: 15.082316993566508
    ------
    Trial # 1
     n= 9, k=2
    cost_matrix = 
    [[None, 3.101934938999485, 4.317358425007676, 4.775976211956362, 0.20883792382206334, 1.9614586146283697, 2.710153732206474, 1.5380331350668919, 0.81989734386962], [4.44036092767616, None, 1.3730600810379234, 2.7484430265188853, 4.356504625652574, 1.5616296313985227, 0.37753607203672046, 1.081979168345622, 0.003966752497756576], [0.08514757191510236, 0.379573464646234, None, 0.8566636004195483, 1.2115456184562434, 4.416202854430882, 2.4750801065001253, 1.9822378394031492, 3.1915402821833534], [2.0058679369147407, 3.1199425760392376, 3.0789095414823118, None, 1.236890761960447, 4.304818578211781, 4.487231604989347, 4.589231967894867, 3.806324193297845], [0.4438069372079667, 2.4483660229856725, 1.9547457033739968, 3.3814889479263908, None, 3.6738419494803267, 3.3704401030482005, 4.3701787996644565, 2.4719079304912794], [2.066037481872167, 3.6317753785477507, 4.881027763879803, 1.915953671790126, 4.014216742987237, None, 4.628399192235907, 4.15856113612508, 2.230539319987092], [2.115100171659722, 1.5801295806826732, 4.260078490526926, 0.8347281218914621, 2.273152474786142, 4.691323748382533, None, 3.208169300564614, 3.7259242727387667], [2.511055728128841, 1.0714751092944468, 0.40781417391318076, 3.2424574528993073, 0.1868517244892942, 0.42598216761702723, 1.1201769689344743, None, 2.2652226199654986], [2.206694097229854, 1.334662312351318, 2.015183872016604, 1.8777644970570768, 1.2059259512237852, 1.5772191688961117, 1.9666806244797286, 3.7949042022748065, None]]
    Your code returned tours: [[0, 8, 5], [0, 7, 2, 1, 6, 3, 4]]
    Tour cost obtained by your code: 9.681536661360802
    ------
    Trial # 2
     n= 11, k=2
    cost_matrix = 
    [[None, 2.2296504391701633, 3.742585610471829, 0.430361983263684, 3.2256777830525083, 2.9140351228590227, 4.981777568461474, 2.3251248231691384, 3.7296239121697488, 3.340371609245392, 1.485119527886432], [0.3928966478819951, None, 2.853267743991768, 3.742436116476574, 0.12264178946310378, 4.174194844772268, 1.6064529614050467, 1.5281277326256537, 3.1999193142027145, 1.1304732402716784, 1.0397861300693463], [2.478386806672024, 4.404529068309778, None, 0.9669755148408393, 0.5752265281230817, 4.22859345235396, 4.3046571982414825, 1.5664970202956745, 2.167041784561218, 3.3771066026709744, 4.036491561533689], [2.8409820402325217, 2.587462499755147, 3.6688170470749886, None, 2.4063049393080598, 1.7275713831565742, 3.1971110314321773, 0.5072445740658021, 0.8812839896965596, 3.417160811579278, 2.6026300113307776], [2.591721851254534, 3.135201519514743, 2.8268969449031136, 4.4821591219274275, None, 2.5438960447874868, 4.876023041904648, 1.7665233502308464, 4.631034149611563, 3.5815396360969283, 4.835374791618971], [0.08850824596329576, 1.4506679964917268, 3.1530089297026698, 4.265874134339276, 3.229880863924418, None, 3.6415351983467814, 3.919062999888607, 0.8007339674410918, 3.510326269403392, 1.5241883157624647], [3.0336586135460317, 0.7249698981327918, 2.3009906075868103, 1.7905804118384512, 3.748750446646869, 1.7125524291751448, None, 3.532887151839079, 1.2675003128359619, 4.895772229105255, 3.1574465293555227], [2.409378097867049, 3.695025726239466, 4.796867696360797, 2.435436518935974, 0.42569123762854655, 3.335940662244231, 1.4898620843462167, None, 0.6923211799963747, 3.6171026924762195, 1.0188951527165924], [0.41292584535196963, 2.9740225949748322, 0.7107623349714842, 3.09451716795499, 3.03152024054182, 1.5014950563928449, 0.9521072739085334, 3.8881133262082894, None, 1.479211859236909, 4.961212046617785], [2.703194205076891, 3.7343140758711693, 1.0297125051512217, 1.9686101014286166, 0.5071525869970006, 3.932285405919868, 1.2808554518961874, 0.3292253848035459, 1.3167187974455268, None, 3.836480833555224], [4.869542169191429, 4.152735698977016, 0.5199696524290293, 0.11518372803260235, 2.896018066379173, 3.6278341695807983, 3.4449502761526003, 1.5704421960348562, 3.3259729518820706, 1.6517945847554523, None]]
    Your code returned tours: [[0, 3, 7, 8, 9, 6, 1], [0, 10, 2, 4, 5]]
    Tour cost obtained by your code: 10.720581593663068
    ------
    Trial # 3
     n= 7, k=3
    cost_matrix = 
    [[None, 4.616559595677063, 3.108938642845416, 0.9423414999519825, 4.798050670768482, 0.6578872886129317, 4.296464140353045], [1.9694063300254439, None, 3.040672518669527, 3.8175410861314103, 4.643873603120988, 4.846049780328646, 1.541756348531097], [2.8829737466510483, 0.8159717996894034, None, 1.6713340733798887, 2.6860764119091596, 1.3664813442219619, 4.115741271374439], [4.404323623379083, 1.917989387372618, 1.1614512607076737, None, 3.3435373509574626, 2.875097077159681, 1.9059762225009473], [2.7600331505482827, 2.721128568405044, 1.0689177529679683, 2.529172704698551, None, 1.0356932922022366, 2.111387480081732], [1.6095074731391374, 0.21191111997994727, 3.408207568830423, 2.07358510204662, 4.740179924814801, None, 0.6108023130073548], [1.6445034783575296, 2.2166327716300085, 2.6147461451789518, 1.9435266064638608, 1.9885060806540755, 2.8332884006401846, None]]
    Your code returned tours: [[0, 5, 6], [0, 3, 2, 1], [0, 4]]
    Tour cost obtained by your code: 15.360447791669085
    ------
    Trial # 4
     n= 10, k=4
    cost_matrix = 
    [[None, 1.2362576282827868, 0.42338314317316494, 1.6194396430239355, 2.180277955080421, 3.953858971349809, 3.316442540053364, 4.103947752593204, 0.8280156834135127, 1.646005738543559], [0.930703173115664, None, 2.859109220047083, 3.429206146871148, 1.1657337033983983, 3.301182256483618, 4.559975841606277, 3.56431797316628, 4.003202995021419, 0.4000768422123102], [0.9296760485409494, 0.5459561738198188, None, 0.5314696875530611, 3.422623818969245, 2.1533462303856976, 0.6789216106571161, 3.724403517219628, 1.5489191297081717, 2.1716426335213956], [1.9511464816699042, 4.3097339853387915, 1.165948155163038, None, 0.07778978766817046, 3.5546004292914626, 1.7481997315052777, 2.544602186543965, 2.5408694714689233, 1.9131621256532472], [2.2653230296342937, 1.0338824014307046, 2.803814566739551, 3.251745129068466, None, 2.3837402151198592, 2.919990812916431, 1.90588733407757, 4.03671100008469, 3.6524874506441085], [4.74682937032836, 0.3584506812994048, 1.3643582063528048, 1.7798005833044916, 0.7218217089383078, None, 3.7305451579425117, 0.3491765259048457, 2.098434057828536, 4.124284626570559], [0.1902789256924936, 1.1577547308249458, 2.5444911761453715, 0.14276326991320143, 0.6914455851109685, 1.7102946929813323, None, 4.658927186158483, 1.2650233415947598, 0.3701238653473016], [2.6894400738111877, 3.4869741515288943, 3.5799328023643806, 3.0500951990794496, 0.959623440557868, 2.2562952775284604, 4.511207657548799, None, 2.4931123382246927, 0.9928880834904458], [2.8789954106487077, 1.4314151312095746, 3.626315483165188, 1.1070417256289766, 1.02717308185375, 4.685125486291227, 3.711994609254896, 0.8319909588544822, None, 3.294751044557426], [1.4967852082386708, 3.666515682589865, 1.1453510601451318, 4.323866345213424, 4.901793109444358, 0.5453242921399909, 1.6842098788707838, 3.6980494919956253, 4.833545415949461, None]]
    Your code returned tours: [[0, 2, 6], [0, 8, 3, 4], [0, 1], [0, 9, 5, 7]]
    Tour cost obtained by your code: 12.967661337665762
    ------
    test passed: 15 points


## Problem 1 B

Notice that in previous part, it happens that with $k=4$ salespeople, we actually get a worse cost than using $k=3$ people. You can try out a few examples to convince yourself as to why this happens. 

We wish to modify the problem to allow salespeople to idle. In other words, although we input $k$ salespeople, the tour we construct may involve $1 \leq l \leq k$ salespeople. 

Modify the ILP formulation from the previous problem to solve the problem of up to $k$ people rather than exactly $k$ salespeople. Note that we still require that every vertex be visited exactly once by some salesperson. 

Complete the implementation of the function `upto_k_tsp_mtz_encoding(n, k, cost_matrix)` below. It follows the same input convention as previous problem but note that we are now computing a tour with at most $k$ salespeople. In other words, not all salespeople need be employed in the tour.

Your code must return a list `lst` that has less than or equal to  $k$ lists, wherein `lst[j]` represents the locations visited by the $j^{th}$ salesperson. 

For Example-1 from the previous part above, for $k=2$ or $k=3$, your code must return 
~~~
[ [0, 3, 1, 4, 2] ]
~~~
As it turns out, in this example a single salesperson suffices to yield optimal cost.


```python
from pulp import *

def upto_k_tsp_mtz_encoding(n, k, cost_matrix):
    # check inputs are OK
    assert 1 <= k < n
    assert len(cost_matrix) == n, f'Cost matrix is not {n}x{n}'
    assert all(len(cj) == n for cj in cost_matrix), f'Cost matrix is not {n}x{n}'

    prob = LpProblem('kTSP', LpMinimize)

    # Decision variables
    x = {(i, j): LpVariable(f'x_{i}_{j}', cat='Binary') for i in range(n) for j in range(n) if i != j}
    t = {i: LpVariable(f't_{i}', lowBound=1, upBound=n, cat='Integer') for i in range(1, n)}

    # Objective function
    prob += sum(cost_matrix[i][j] * x[(i, j)] for i in range(n) for j in range(n) if i != j and cost_matrix[i][j] is not None)

    # Degree constraints
    for i in range(n):
        if i == 0:
            prob += sum(x[(0, j)] for j in range(1, n) if cost_matrix[0][j] is not None) <= k
            prob += sum(x[(j, 0)] for j in range(1, n) if cost_matrix[j][0] is not None) <= k
        else:
            prob += sum(x[(i, j)] for j in range(n) if j != i and cost_matrix[i][j] is not None) == 1
            prob += sum(x[(j, i)] for j in range(n) if j != i and cost_matrix[j][i] is not None) == 1

    # Time stamp constraints
    for i in range(1, n):
        for j in range(1, n):
            if i != j and cost_matrix[i][j] is not None:
                prob += t[i] - t[j] + (n * x[(i, j)]) <= n - 1

    # Solve the problem
    prob.solve()

    # Construct the solution
    solution = []
    visited_nodes = set()
    current_salesperson = 0
    current_node = 0

    solution.append([current_node])
    visited_nodes.add(current_node)

    while len(visited_nodes) < n:
        next_node = None
        min_cost = float('inf')

        for j in range(1, n):
            if cost_matrix[current_node][j] is not None and x[(current_node, j)].value() > 0.5 and j not in visited_nodes:
                if cost_matrix[current_node][j] < min_cost:
                    next_node = j
                    min_cost = cost_matrix[current_node][j]

        if next_node is not None:
            current_node = next_node
            solution[current_salesperson].append(current_node)
            visited_nodes.add(current_node)
        else:
            # If no more edges can be found, start a new tour or stop
            if len(visited_nodes) == n:
                break
            current_salesperson += 1
            current_node = 0
            solution.append([current_node])
            visited_nodes.add(current_node)

    return solution
```


```python
cost_matrix=[ [None,3,4,3,5],
             [1, None, 2,4, 1],
             [2, 1, None, 5, 4],
             [1, 1, 5, None, 4],
             [2, 1, 3, 5, None] ]
n=5
k=3
all_tours = upto_k_tsp_mtz_encoding(n, k, cost_matrix)
print(f'Your code returned tours: {all_tours}')
assert len(all_tours) <= k, f'<= {k} tours -- your code returns {len(all_tours)} tours instead'

tour_cost = 0
for tour in all_tours:
    assert tour[0] == 0, 'Each salesperson tour must start from vertex 0'
    i = 0
    for j in tour[1:]:
        tour_cost += cost_matrix[i][j]
        i = j
    tour_cost += cost_matrix[i][0]

assert len(all_tours) == 1, f'In this example, just one salesperson is needed to optimally visit all vertices. Your code returns {len(all_tours)}'
print(f'Tour cost obtained by your code: {tour_cost}')
assert abs(tour_cost - 10) <= 0.001, f'Expected tour cost is 10, your code returned {tour_cost}'
for i in range(1, n):
    is_in_tour = [ 1 if i in tour else 0 for tour in all_tours]
    assert sum(is_in_tour) == 1, f' vertex {i} is in {sum(is_in_tour)} tours -- this is incorrect'

print('test passed: 3 points')
```

    Your code returned tours: [[0, 3, 1, 4, 2]]
    Tour cost obtained by your code: 10
    test passed: 3 points



```python
cost_matrix = [ 
 [None, 1, 1, 1, 1, 1, 1, 1],
    [0, None, 1, 2, 1, 1, 1, 1],
    [1, 0, None, 1, 2, 2, 2, 1],
    [1, 2, 2, None, 0, 1, 2, 1],
    [1, 1, 1, 1, None, 1, 1, 1],
    [0,  1, 2, 1, 1, None, 1, 1],
    [1, 0,  1, 2, 2, 2,None, 1],
    [1, 2, 2, 0, 1, 2, 1, None],
]
n = 8
k = 5

all_tours = upto_k_tsp_mtz_encoding(n, k, cost_matrix)
print(f'Your code returned tours: {all_tours}')
assert len(all_tours) <= k, f'k={k} must yield two tours -- your code returns {len(all_tours)} tours instead'

tour_cost = 0
for tour in all_tours:
    assert tour[0] == 0, 'Each salesperson tour must start from vertex 0'
    i = 0
    for j in tour[1:]:
        tour_cost += cost_matrix[i][j]
        i = j
    tour_cost += cost_matrix[i][0]

print(f'Tour cost obtained by your code: {tour_cost}')
assert abs(tour_cost - 4) <= 0.001, f'Expected tour cost is 4, your code returned {tour_cost}'
for i in range(1, n):
    is_in_tour = [ 1 if i in tour else 0 for tour in all_tours]
    assert sum(is_in_tour) == 1, f' vertex {i} is in {sum(is_in_tour)} tours -- this is incorrect'

print('test passed: 3 points')
```

    Your code returned tours: [[0, 6, 2, 1], [0, 7, 3, 4, 5]]
    Tour cost obtained by your code: 4
    test passed: 3 points



```python
Problem 1 Bfrom random import uniform, randint

def create_cost(n):
    return [ [uniform(0, 5) if i != j else None for j in range(n)] for i in range(n)]

for trial in range(20):
    print(f'Trial # {trial}')
    n = randint(5, 11)
    k = randint(2, n//2)
    print(f' n= {n}, k={k}')
    cost_matrix = create_cost(n)
    print('cost_matrix = ')
    print(cost_matrix)
    all_tours = upto_k_tsp_mtz_encoding(n, k, cost_matrix)
    print(f'Your code returned tours: {all_tours}')
    assert len(all_tours) <= k, f'k={k} must yield two tours -- your code returns {len(all_tours)} tours instead'

    tour_cost = 0
    for tour in all_tours:
        assert tour[0] == 0, 'Each salesperson tour must start from vertex 0'
        i = 0
        for j in tour[1:]:
            tour_cost += cost_matrix[i][j]
            i = j
        tour_cost += cost_matrix[i][0]

    print(f'Tour cost obtained by your code: {tour_cost}')
    #assert abs(tour_cost - 6) <= 0.001, f'Expected tour cost is 6, your code returned {tour_cost}'
    for i in range(1, n):
        is_in_tour = [ 1 if i in tour else 0 for tour in all_tours]
        assert sum(is_in_tour) == 1, f' vertex {i} is in {sum(is_in_tour)} tours -- this is incorrect'
    print('------')
print('test passed: 4 points')
```


      File "<ipython-input-43-a560a206289f>", line 1
        Problem 1 Bfrom random import uniform, randint
                ^
    SyntaxError: invalid syntax



## Problem 2 (10 points)

We noted the use of Christofides algorithm for metric TSP. We noted that for non-metric TSPs it does not work. 
In fact, the shortcutting used in Christofides algorithm can be _arbitrarily_ bad for a TSP that is symmetric but fails to be a metric TSP.

In this example, we would like you to frame a symmetric TSP instance ($C_{ij} = C_{ji}$) with $5$ vertices wherein the algorithm obtained by "shortcutting" the minimum spanning tree (MST), that would be a 2-factor approximation for metric TSP, yields an answer that can be quite "far off" from the optimal solution.

Enter a __symmetric__ cost-matrix for the TSP below as a 5x5 matrix as a list of lists following convention in our notes. such that the optimal answer is at least $10^6$ times smaller than that obtained by the TSP-based approximation. We will test your answer by running the TSP with shortcutting algorithm.

__Hint:__ Force the edges $(0,1), (1,2), (2,3)$ and $(3,4)$ to be the minimum spanning tree. But make the weight of the edge form $4$ back to $0$ very high.


__Note:__ this problem is tricky and requires you to be very familiar with how Christofides algorithm works. It may be wise to attempt the remaining problems first before this one. Do not worry about the diagonal entry of your matrices.



```python
cost_matrix = [
    [None, 1, 1, 1, 10**9],
    [1, None, 1, 1, 1],
    [1, 1, None, 1, 1],
    [1, 1, 1, None, 1],
    [10**9, 1, 1, 1, None]
]
```


```python
# check that the cost matrix is symmetric.
assert len(cost_matrix) == 5, f'Cost matrix must have 5 rows. Yours has {len(cost_matrix)} rows'
assert all(len(cj) == 5 for cj in cost_matrix), f'Each row of the cost matrix must have 5 entries.'
for i in range(5):
    for j in range(i):
        assert cost_matrix[i][j] == cost_matrix[j][i], f'Cost matrix fails to be symmetric at entries {(i,j)} and {(j,i)}'
print('Structure of your cost matrix looks OK (3 points).')
```

    Structure of your cost matrix looks OK (3 points).


Please ensure that you run the two cells below or else, your tests will fail.


```python
# MST based tsp approximation
import networkx as nx

# This code implements the simple MST based shortcutting approach that would yield factor of 2
# approximation for metric TSPs.
def minimum_spanning_tree_tsp(n, cost_matrix):
    G = nx.Graph()
    for i in range(n):
        for j in range(i):
            G.add_edge(i, j, weight=cost_matrix[i][j])
    T = nx.minimum_spanning_tree(G)
    print(f'MST for your graph has the edges {T.edges}')
    mst_cost = 0
    mst_dict = {} # store mst as a dictionary
    for (i,j) in T.edges:
        mst_cost += cost_matrix[i][j]
        if i in mst_dict:
            mst_dict[i].append(j)
        else:
            mst_dict[i] = [j]
        if j in mst_dict:
            mst_dict[j].append(i)
        else:
            mst_dict[j] = [i]
    print(f'MST cost: {mst_cost}')
    print(mst_dict)
    # Let's form a tour with short cutting
    def traverse_mst(tour_so_far, cur_node):
        assert cur_node in mst_dict
        next_nodes = mst_dict[cur_node]
        for j in next_nodes:
            if j in tour_so_far:
                continue
            tour_so_far.append(j)
            traverse_mst(tour_so_far, j)
        return
    tour = [0]
    traverse_mst(tour, 0)
    i = 0
    tour_cost = 0
    for j in tour[1:]:
        tour_cost += cost_matrix[i][j]
        i = j
    tour_cost += cost_matrix[i][0]
    return tour, tour_cost
```


```python
# optimal TSP tour taken from our notes using MTZ encoding
from pulp import *

def mtz_encoding_tsp(n, cost_matrix):
    assert len(cost_matrix) == n, f'Cost matrix is not {n}x{n}'
    assert all(len(cj) == n for cj in cost_matrix), f'Cost matrix is not {n}x{n}'
    # create our encoding variables
    binary_vars = [ # add a binary variable x_{ij} if i not = j else simply add None
        [ LpVariable(f'x_{i}_{j}', cat='Binary') if i != j else None for j in range(n)] 
        for i in range(n) ]
    # add time stamps for ranges 1 .. n (skip vertex 0 for timestamps)
    time_stamps = [LpVariable(f't_{j}', lowBound=0, upBound=n, cat='Continuous') for j in range(1, n)]
    # create the problem
    prob = LpProblem('TSP-MTZ', LpMinimize)
    # create add the objective function 
    objective_function = lpSum( [ lpSum([xij*cj if xij != None else 0 for (xij, cj) in zip(brow, crow) ])
                           for (brow, crow) in zip(binary_vars, cost_matrix)] )
    
    prob += objective_function 
    
    # add the degree constraints
    for i in range(n):
        # Exactly one leaving variable
        prob += lpSum([xj for xj in binary_vars[i] if xj != None]) == 1
        # Exactly one entering
        prob += lpSum([binary_vars[j][i] for j in range(n) if j != i]) == 1
    # add time stamp constraints
    for i in range(1,n):
        for j in range(1, n):
            if i == j: 
                continue
            xij = binary_vars[i][j]
            ti = time_stamps[i-1]
            tj = time_stamps[j -1]
            prob += tj >= ti + xij - (1-xij)*(n+1) # add the constraint
    # Done: solve the problem
    status = prob.solve(PULP_CBC_CMD(msg=False)) # turn off messages
    assert status == constants.LpStatusOptimal, f'Unexpected non-optimal status {status}'
    # Extract the tour
    tour = [0]
    tour_cost = 0
    while len(tour) < n:
        i = tour[-1]
        # find all indices j such that x_ij >= 0.999 
        sols = [j for (j, xij) in enumerate(binary_vars[i]) if xij != None and xij.varValue >= 0.999]
        assert len(sols) == 1, f'{sols}' # there better be just one such vertex or something has gone quite wrong
        j = sols[0] # extract the lone solutio 
        tour_cost = tour_cost + cost_matrix[i][j] # add to the tour cost
        tour.append(j) # append to the tour
        assert j != 0
    i = tour[-1]
    tour_cost = tour_cost + cost_matrix[i][0]
    return tour, tour_cost
        
```


```python
#test that exact answer is 10^6 times smaller than approximate answer.
# compute MST based approximation
tour, tour_cost = minimum_spanning_tree_tsp(5, cost_matrix)
print(f'MST approximation yields tour is {tour} with cost {tour_cost}')
# compute exact answer
opt_tour, opt_tour_cost = mtz_encoding_tsp(5, cost_matrix)
print(f'Optimal tour is {opt_tour} with cost {opt_tour_cost}')
# check that the fraction is 1million times apart.
assert tour_cost/opt_tour_cost >= 1E+06, 'The TSP + shortcutting tour must be at least 10^6 times costlier than optimum. In your case, the ratio is {tour_cost/opt_tour_cost}'
print('Test passed: 7 points')
```

    MST for your graph has the edges [(1, 0), (1, 2), (1, 3), (1, 4)]
    MST cost: 4
    {1: [0, 2, 3, 4], 0: [1], 2: [1], 3: [1], 4: [1]}
    MST approximation yields tour is [0, 1, 2, 3, 4] with cost 1000000004
    Optimal tour is [0, 1, 4, 3, 2] with cost 5
    Test passed: 7 points


## Problem 3

In this problem, we wish to solve TSP with additional constraints. Suppose we are given a TSP instance in the form of a $n\times n$ matrix $C$ representing a complete graph. 

We wish to solve a TSP but with additional constraints specified as a list $[(i_0, j_0), \ldots, (i_k, j_k)]$ wherein each pair $(i_l, j_l)$ in the list specifies that vertex $i_l$ must be visited in the tour before vertex $j_l$. Assume that the tour starts/ends at vertex $0$ and none of the vertices in the constraint list is $0$. I.e, $i_l\not= 0, j_l \not= 0$ for all $0 \leq l \leq k$.

Modify one of the ILP encodings we have presented to solve TSP with extra constraints. Implement your solution in the function `tsp_with_extra_constraints(n, cost_matrix, constr_list)` where the extra argument `constr_list` is a list of pairs `[(i0,j0),...., (ik, jk)]` that specify for each pair `(il,jl)` that vertex `il` must be visited before `jl`. Assume that the problem is feasible (no need to handle infeasible instances). 
Your code should output the optimal tour as a list.

## Example

Consider again the graph with $5$ nodes and the following cost matrix from problem 1:

$$ \begin{array}{c|ccccc}
  \text{Vertices} & 0 & 1 & 2 & 3 & 4 \\ 
   \hline
 0 & - & 3 & 4 & 3 & 5 \\ 
 1 & 1 & - & 2 & 4 & 1 \\ 
 2 & 2 & 1 & - & 5 & 4 \\ 
 3 & 1 & 1 & 5 & - & 4 \\ 
 4 & 2 & 1 & 3 & 5 & - \\ 
 \end{array}$$
 
The optimal TSP tour will be $[0, 3, 1, 4, 2]$ with total cost $10$.

Suppose we added the constraints $[(4, 3), (1, 2)]$ we note that the tour satisfies the constraint $(1, 2)$ since it visits vertex $1$ before vertex $2$ but it unfortunately, $(4,3)$ is violated since vetex $3$ is visited before $4$ in the tour.



```python
from pulp import LpProblem, LpMinimize, LpVariable, LpInteger, lpSum, value

def tsp_with_extra_constraints(n, cost_matrix, constraints):
    assert len(cost_matrix) == n, f'Cost matrix is not {n}x{n}'
    assert all(len(cj) == n for cj in cost_matrix), f'Cost matrix is not {n}x{n}'
    assert all(1 <= i < n and 1 <= j < n and i != j for (i, j) in constraints)

    # Create the problem
    prob = LpProblem("TSP with Constraints", LpMinimize)

    # Create decision variables
    x = {(i, j): LpVariable(f"x_{i}_{j}", cat=LpInteger, lowBound=0, upBound=1)
         for i in range(n) for j in range(n) if i != j}
    u = {i: LpVariable(f"u_{i}", cat=LpInteger, lowBound=2, upBound=n)
         for i in range(1, n)}

    # Objective function
    prob += lpSum(cost_matrix[i][j] * x[i, j] for i in range(n) for j in range(n) if i != j)

    # Constraints
    # Each node is visited exactly once
    for i in range(n):
        prob += lpSum(x[i, j] for j in range(n) if j != i) == 1
        prob += lpSum(x[j, i] for j in range(n) if j != i) == 1

    # Subtour elimination constraints
    for i in range(1, n):
        for j in range(1, n):
            if i != j:
                prob += u[i] - u[j] + (n - 1) * x[i - 1, j - 1] <= n - 2

    # Additional constraints: vertex i must be visited before vertex j
    for i, j in constraints:
        prob += u[i] - u[j] + (n - 1) * (1 - x[i - 1, j - 1]) <= n - 2

    # Solve the problem
    prob.solve()

    # Extract the solution
    tour = [0]
    cur_node = 0
    for _ in range(n - 1):
        for j in range(n):
            if j != cur_node and value(x[cur_node, j]) == 1:
                tour.append(j)
                cur_node = j
                break

    return tour
```


```python
cost_matrix=[ [None,3,4,3,5],
             [1, None, 2,4, 1],
             [2, 1, None, 5, 4],
             [1, 1, 5, None, 4],
             [2, 1, 3, 5, None] ]
n=5
constraints = [(3,4),(1,2)]
tour = tsp_with_extra_constraints(n, cost_matrix, constraints)
i = 0
tour_cost = 0
for j in tour[1:]:
    tour_cost += cost_matrix[i][j]
    i = j
tour_cost += cost_matrix[i][0]
print(f'Tour:{tour}')
print(f'Cost of your tour: {tour_cost}')
assert abs(tour_cost-10) <= 0.001, 'Expected cost was 10'
for i in range(n):
    num = sum([1 if j == i else 0 for j in tour])
    assert  num == 1, f'Vertex {i} repeats {num} times in tour'
for (i, j) in constraints:
    assert tour.index(i) < tour.index(j), f'Tour does not respect constraint {(i,j)}'
print('Test Passed (3 points)')
```

    Tour:[0, 3, 1, 4, 2]
    Cost of your tour: 10
    Test Passed (3 points)



```python
cost_matrix=[ [None,3,4,3,5],
             [1, None, 2,4, 1],
             [2, 1, None, 5, 4],
             [1, 1, 5, None, 4],
             [2, 1, 3, 5, None] ]
n=5
constraints = [(4,3),(1,2)]
tour = tsp_with_extra_constraints(n, cost_matrix, constraints)
i = 0
tour_cost = 0
for j in tour[1:]:
    tour_cost += cost_matrix[i][j]
    i = j
tour_cost += cost_matrix[i][0]
print(f'Tour:{tour}')
print(f'Cost of your tour: {tour_cost}')
assert abs(tour_cost-13) <= 0.001, 'Expected cost was 13'
for i in range(n):
    num = sum([1 if j == i else 0 for j in tour])
    assert  num == 1, f'Vertex {i} repeats {num} times in tour'
for (i, j) in constraints:
    assert tour.index(i) < tour.index(j), f'Tour does not respect constraint {(i,j)}'
print('Test Passed (3 points)')
```

    Tour:[0, 2, 1, 4, 3]
    Cost of your tour: 12



    ---------------------------------------------------------------------------

    AssertionError                            Traceback (most recent call last)

    <ipython-input-67-a0da6aa74d9e> in <module>
         15 print(f'Tour:{tour}')
         16 print(f'Cost of your tour: {tour_cost}')
    ---> 17 assert abs(tour_cost-13) <= 0.001, 'Expected cost was 13'
         18 for i in range(n):
         19     num = sum([1 if j == i else 0 for j in tour])


    AssertionError: Expected cost was 13



```python
from random import uniform, randint

def create_cost(n):
    return [ [uniform(0, 5) if i != j else None for j in range(n)] for i in range(n)]

for trial in range(20):
    print(f'Trial # {trial}')
    n = randint(6, 11)
    cost_matrix = create_cost(n)
    constraints = [(1, 3), (4, 2), (n-1, 1), (n-2, 2)]
    tour = tsp_with_extra_constraints(n, cost_matrix, constraints)
    i = 0
    tour_cost = 0
    for j in tour[1:]:
        tour_cost += cost_matrix[i][j]
        i = j
    tour_cost += cost_matrix[i][0]
    print(f'Tour:{tour}')
    print(f'Cost of your tour: {tour_cost}')
    for i in range(n):
        num = sum([1 if j == i else 0 for j in tour])
        assert  num == 1, f'Vertex {i} repeats {num} times in tour'
    for (i, j) in constraints:
        assert tour.index(i) < tour.index(j), f'Tour does not respect constraint {(i,j)}'
print('Test Passed (10 points)')
```

    Trial # 0
    Tour:[0, 2, 4, 7, 1, 9, 6, 8, 3, 5]
    Cost of your tour: 6.427152156015268



    ---------------------------------------------------------------------------

    AssertionError                            Traceback (most recent call last)

    <ipython-input-68-2eade90c209d> in <module>
         22         assert  num == 1, f'Vertex {i} repeats {num} times in tour'
         23     for (i, j) in constraints:
    ---> 24         assert tour.index(i) < tour.index(j), f'Tour does not respect constraint {(i,j)}'
         25 print('Test Passed (10 points)')


    AssertionError: Tour does not respect constraint (4, 2)


## Answers to Select Problems

### 1A part A

- Vertex 0: $k$ edges leave and $k$ edges enter.
- Vertex 1, ..., n-1: 1 edge leaves and 1 edge enters (same as TSP).

### 1A part B

This is a trick question. There is no need to change any of the time stamp related constraints.


## That's All Folks!
