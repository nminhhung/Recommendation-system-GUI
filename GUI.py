import streamlit as st
from streamlit_pandas_profiling import st_profile_report
st.set_option('deprecation.showPyplotGlobalUse', False)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
from ydata_profiling import ProfileReport
import plotly.figure_factory as ff
import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.metrics.pairwise import cosine_similarity
import time
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from PIL import Image

import findspark
findspark.init()
from pyspark.sql import SparkSession
spark = SparkSession.builder.master("local[*]").getOrCreate()
# Check the pyspark version
import pyspark
from pyspark.ml.feature import StringIndexer
from pyspark.ml.recommendation import ALS
from pyspark.sql.functions import col
import time
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.recommendation import ALSModel
from pyspark.sql.functions import explode
import base64
from PIL import Image
import base64

# Loading Image using PIL
im = Image.open('img/anya.jpg')
# Adding Image to web app
st.set_page_config(page_title="Recommendation Systems App", page_icon = im)

import gc
# Enable garbage collection
gc.enable()

def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/jpg;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)
set_background('img/bg2.jpg')






# Top GUI


st.title('nminhhung Blogspot')






# audio_file = open('img/bgm.mp3', 'rb')
# audio_bytes = audio_file.read()

# audio_code = f'''
# <audio controls autoplay>
#   <source src="data:audio/mp3;base64,{base64.b64encode(audio_bytes).decode()}" type="audio/mp3">
# </audio>
# '''

# audio_minimized_code = '''
# <style>
# audio {
#   width: 200px;
#   height: 40px;
#   position: fixed;
#   top: 50px;
#   right: 10px;
#   z-index: 9999;
# }
# </style>
# '''

# st.markdown(audio_code + audio_minimized_code, unsafe_allow_html=True)



audio_file = open('img/bgm.mp3', 'rb')
audio_bytes = audio_file.read()

audio_code = f'''
<audio id="audio-player" controls autoplay>
  <source src="data:audio/mp3;base64,{base64.b64encode(audio_bytes).decode()}" type="audio/mpeg">
</audio>
'''

audio_minimized_code = '''
<style>
audio {
  width: 200px;
  height: 40px;
  position: fixed;
  top: 50px;
  right: 10px;
  z-index: 9999;
}
</style>
'''

play_audio_script = '''
<script>
var audio = document.getElementById("audio-player");
audio.play();
</script>
'''

st.markdown(audio_code + audio_minimized_code + play_audio_script, unsafe_allow_html=True)





st.write("""
## Recommendation Systems
""")


# Upload file
uploaded_file = st.file_uploader("Choose a file", type=['csv'])
if uploaded_file is not None:
    products = pd.read_csv(uploaded_file, encoding='utf-8')
    products.to_csv("products_new.csv", index = False)

menu = ["Recommendation Systems", "Products EDA", "Reviews EDA", "Content Base Filtering","Collaborative Filtering", "Makes Recommendations"]
choice = st.sidebar.selectbox('Menu', menu)
st.write("You've chosen: ", choice)

image2 = Image.open("im2.png")
st.sidebar.image(image2, caption="Recommendation Systems", use_column_width=True)
image4 = Image.open("im4.png")
st.sidebar.image(image4, caption="Engine", use_column_width=True)
image3 = Image.open("im3.png")
st.sidebar.image(image3, caption="Applicability", use_column_width=True)
image5 = Image.open("img/im5.png")
st.sidebar.image(image5, caption="Reality", use_column_width=True)


st.markdown(
    """
    <style>
    .sidebar .sidebar-content {
        background-color: #F5F5F5;
        padding: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)




# Function:

def eda(df):
    # Print the shape of the dataframe
    st.write('Shape of the data:', df.shape)

    # Print the summary statistics of the numerical columns
    st.write('Summary statistics of the numerical columns:')
    st.write(df.describe())

    # Display the column names, data types, and non-null value counts for each column
    st.write('Column names and data types:')
    st.write(df.dtypes)
    st.write('Number of non-null values per column:')
    st.write(df.count())

    # Select only the numerical columns and drop any rows with missing values
    num_cols = df.select_dtypes(include='number').dropna()

    # Plot a histogram of the numerical columns
    fig1, ax = plt.subplots(figsize=(15, 10))
    ax.hist(num_cols, bins=10)
    ax.set_title('Histogram of numerical columns')
    st.pyplot(fig1)

    # Plot a correlation matrix of the numerical columns
    fig2, ax = plt.subplots(figsize=(15, 10))
    sns.heatmap(num_cols.corr(), annot=True, cmap='coolwarm')
    ax.set_title('Correlation matrix of numerical columns')
    st.pyplot(fig2)

    # Plot a boxplot of the numerical columns
    fig3, ax = plt.subplots(figsize=(15, 10))
    num_cols.plot(kind='box', subplots=True, layout=(3,3), sharex=False, sharey=False, ax=ax)
    ax.set_title('Boxplot of numerical columns')
    st.pyplot(fig3)

    # Plot a pairplot of the numerical columns
    fig4 = sns.pairplot(num_cols, diag_kind='kde')
    fig4.fig.suptitle('Pairplot of numerical columns')
    st.pyplot(fig4.fig)


def perform_eda(products):
    # Visualize the distribution of ratings
    fig, ax = plt.subplots()
    sns.distplot(products['rating'], kde=True, color='blue', ax=ax)
    ax.set(title='Distribution of Ratings', xlabel='Rating')
    st.pyplot(fig)

    # Visualize the relationship between price and rating
    fig, ax = plt.subplots()
    sns.scatterplot(data=products, x='rating', y='price', color='green', ax=ax)
    ax.set(title='Price vs. Rating', xlabel='Rating', ylabel='Price')
    st.pyplot(fig)

    # Visualize the count of products in each brand
    fig, ax = plt.subplots()
    sns.countplot(data=products, x='brand', color='purple', ax=ax)
    ax.set(title='Count of Products by Brand', xlabel='Brand', ylabel='Count')
    st.pyplot(fig)

    # Create a word cloud of the product Preprocessing
    text = ' '.join(products['products_pre'])
    wordcloud = WordCloud(width=800, height=800, collocations=False, background_color='white', min_font_size=10).generate(text)
    fig, ax = plt.subplots(figsize=(8, 8), facecolor=None)
    ax.imshow(wordcloud)
    ax.axis("off")
    ax.set(title='Word Cloud of Product Description Preprocessing')
    st.pyplot(fig) 

def visualize_products(products):
    # Distribution of ratings
    fig11 = px.histogram(products, x='rating', nbins=20, title='Distribution of ratings')

    # Scatter plot of price vs. rating
    fig22 = px.scatter(products, x='price', y='rating', title='Price vs. Rating')

    # Grouped bar chart of average rating by brand
    brand_rating = products.groupby('brand')['rating'].mean().reset_index()
    fig33 = px.bar(brand_rating, x='brand', y='rating', title='Average rating by brand')

    # Word cloud of product descriptions
    text = ' '.join(products['description'])
    wordcloud = WordCloud(width=800, height=400, collocations=False, background_color='white').generate(text)

    fig44 = plt.figure(figsize=(14, 7))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud of Product Descriptions')
    plt.close()

    return fig11, fig22, fig33, fig44

def visualize_reviews(reviews):
    # Visualize the distribution of the ratings using seaborn
    st.set_option('deprecation.showPyplotGlobalUse', False) # to suppress warning
    sns.countplot(x='rating', data=reviews)
    plt.title('Distribution of Ratings')
    st.pyplot()

    # Distribution of ratings using plotly
    fig = px.histogram(reviews, x='rating', nbins=5, color_discrete_sequence=['#636EFA'])
    fig.update_layout(title_text='Distribution of Ratings')
    st.plotly_chart(fig)



# GUI Body:

if choice == 'Recommendation Systems':    
    st.subheader("Recommendation Systems")
    
    st.markdown("""
            <p style='text-align: justify;'> 
            Recommendation systems are a powerful tool in modern business operations, enabling companies to provide personalized product recommendations to their customers. By leveraging machine learning algorithms, businesses can analyze customer behavior and purchase histories to generate targeted recommendations that increase customer satisfaction and drive sales. These systems can also help companies to optimize their marketing strategies, by identifying trends and patterns in customer data and tailoring their advertising efforts to better meet customer needs. 
            </p>
            <p style='text-align: justify;'> 
            In addition to driving sales and improving customer satisfaction, recommendation systems can also provide valuable insights into the underlying dynamics of consumer behavior. By analyzing patterns in purchase data, businesses can identify which products are most popular, which demographics are most likely to purchase certain items, and how these trends may change over time. This information can be used to inform product development, marketing strategies, and other key business decisions, allowing companies to stay ahead of the curve and remain competitive in a rapidly-evolving marketplace.
            </p>
            <p style='text-align: justify;'> 
            Overall, recommendation systems powered by machine learning represent a powerful tool for businesses seeking to improve customer engagement, drive sales, and gain valuable insights into the behavior of their target audience. As the technology continues to evolve, it is likely that these systems will become even more sophisticated and capable, providing even greater value to businesses of all sizes and industries.
            </p>
            <p style='text-align: justify;'>             
            </p>
            """, unsafe_allow_html=True)

#     st.write("""
#     ###### Recommendation systems are a powerful tool in modern business operations, enabling companies to provide personalized product recommendations to their customers. By leveraging machine learning algorithms, businesses can analyze customer behavior and purchase histories to generate targeted recommendations that increase customer satisfaction and drive sales. These systems can also help companies to optimize their marketing strategies, by identifying trends and patterns in customer data and tailoring their advertising efforts to better meet customer needs.

#     ###### In addition to driving sales and improving customer satisfaction, recommendation systems can also provide valuable insights into the underlying dynamics of consumer behavior. By analyzing patterns in purchase data, businesses can identify which products are most popular, which demographics are most likely to purchase certain items, and how these trends may change over time. This information can be used to inform product development, marketing strategies, and other key business decisions, allowing companies to stay ahead of the curve and remain competitive in a rapidly-evolving marketplace.

#     ###### Overall, recommendation systems powered by machine learning represent a powerful tool for businesses seeking to improve customer engagement, drive sales, and gain valuable insights into the behavior of their target audience. As the technology continues to evolve, it is likely that these systems will become even more sophisticated and capable, providing even greater value to businesses of all sizes and industries.
#     """)    
    st.image("rs.jpg")





elif choice == "Products EDA":
    st.subheader("EDA on Products dataset")
    products = pd.read_csv('Data/products_clean.csv')
    

    st.write("##### 1. Overview data")
    st.dataframe(products.head(10))
    st.dataframe(products.tail(10))

    st.write("##### 2. EDA data") 
    st.write("##### 2.1. EDA with basic libraries (matplotlib, seaborn)")     
    eda(products) 
    perform_eda(products)

    st.write("##### 2.2. EDA with Plotly")
    # Visualize products
    fig11, fig22, fig33, fig44 = visualize_products(products)
    # Display figures
    st.plotly_chart(fig11)
    st.plotly_chart(fig22)
    st.plotly_chart(fig33)
    st.pyplot(fig44)

    st.write("##### 2.3. EDA with Pandas Profiling")
    # Create the pandas profiling report
    report_products = ProfileReport(products, title='Pandas Profiling Report Products', explorative=True)

    # Display the report using streamlit
    st.write('##### Pandas Profiling Report Products')
    st_profile_report(report_products)
    

    st.write("##### 3.Nhan xet")
    st.write("""#####
    - Da phan cac san pham duoc mua co gia tri < 10M. Va deu duoc rating > 3.
    - Rating cho cac san pham da so deu > 3.
    - Cac brand duoc danh gia rating xap xi tuong duong voi nhau. So it duoc muc rating thap ( < 3)
    - Da xu li cac gia tri null/NaN va duplicate nen ty le con ton tai trong dataset la 0%.
    - `rating`: tap trung nhieu o 0 va tu 4 - 5.
    - `price`: min = 7000, max = 62690000, mean = 2764025.3
    - `brand` : nhieu nhat la `OEM` (1115) va `Samsung` (199)
    """)




# ### Reviews

elif choice == "Reviews EDA":
    st.subheader("EDA on Reviews dataset")
    reviews = pd.read_csv('Data/reviews_clean_1.csv', index_col=0)
    

    st.write("##### 1. Overview data")
    st.dataframe(reviews.head(10))
    st.dataframe(reviews.tail(10))

    st.write("##### 2. EDA data")    
    eda(reviews)
    visualize_reviews(reviews)

    st.write("##### 3. EDA with Pandas Profiling")
    # Create the pandas profiling report
    report_reviews = ProfileReport(reviews, title='Pandas Profiling Report Reviews', explorative=True)

    # Display the report using streamlit
    st.write('##### Pandas Profiling Report Reviews')
    st_profile_report(report_reviews)
    

    st.write("##### 4.Nhan xet")
    st.write("""#####
    - Ty le nguoi dung danh gia rating 4, 5 la dong nhat.
    - Co mot so it outliers trong `rating`.
    """)
# END EDA!





elif choice == "Content Base Filtering":

    data = pd.read_csv('Data/products_clean.csv')
    
    stop_words = []
    with open('Data/vietnamese-stopwords.txt', 'r',encoding='utf-8') as f:
        for line in f:
            stop_words.append(line.strip())

    def preprocess(text):
        words = gensim.utils.simple_preprocess(text, deacc=True)
        words = [w for w in words if w not in stop_words]
        return words    

    st.subheader("Content-Based Filtering with Python")
    st.markdown("""
        <p style='text-align: justify;'> 
        - Content-Based Filtering is a technique used in recommendation systems that recommends items to users based on the characteristics of the items they have previously liked. The system analyzes the properties of the items and creates a profile of the user's preferences based on these properties. The system then recommends items that have similar properties to the items that the user has previously liked.
        </p>
        <p style='text-align: justify;'> 
        - To use Content-Based Filtering with Python, you can use the scikit-learn library. Scikit-learn provides several algorithms that can be used for Content-Based Filtering, such as the cosine similarity algorithm. You can create a feature vector for each item in the dataset based on its properties, and then use the cosine similarity algorithm to calculate the similarity between the feature vectors of different items. You can then recommend items to users based on the similarity between the feature vectors of the items they have previously liked and the feature vectors of other items in the dataset.
        </p>
        """, unsafe_allow_html = True)
#     st.write("""
#     - Content-Based Filtering is a technique used in recommendation systems that recommends items to users based on the characteristics of the items they have previously liked. The system analyzes the properties of the items and creates a profile of the user's preferences based on these properties. The system then recommends items that have similar properties to the items that the user has previously liked.
#     - To use Content-Based Filtering with Python, you can use the scikit-learn library. Scikit-learn provides several algorithms that can be used for Content-Based Filtering, such as the cosine similarity algorithm. You can create a feature vector for each item in the dataset based on its properties, and then use the cosine similarity algorithm to calculate the similarity between the feature vectors of different items. You can then recommend items to users based on the similarity between the feature vectors of the items they have previously liked and the feature vectors of other items in the dataset.
    
#     """)
    st.write("#### 1. Gensim")
    st.write("##### 1.1. Coding")
    code1 = """

import gensim
import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import time


# Preprocess the data
stop_words = []
with open('Data/vietnamese-stopwords.txt', 'r',encoding='utf-8') as f:
    for line in f:
        stop_words.append(line.strip())

def preprocess(text):
    words = gensim.utils.simple_preprocess(text, deacc=True)
    words = [w for w in words if w not in stop_words]
    return words

data['description'] = data['description'].apply(preprocess)

# Convert the data to TaggedDocument format
tagged_data = [TaggedDocument(words=doc, tags=[str(i)]) for i, doc in enumerate(data['description'])]

# Train the model
start_time = time.time()
max_epochs = 200
vec_size = 52  # change the vector size to a multiple of 4
alpha = 0.025
min_alpha = 0.00025
min_count = 5
model = Doc2Vec(vector_size=vec_size, alpha=alpha, min_alpha=min_alpha, min_count=min_count, dm=1, epochs=max_epochs)
model.build_vocab(tagged_data)
model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)
print('Training time:', time.time() - start_time)

# Test the model
test_data = ['chuột logitech', 'loa bluetooth', 'tai nghe apple airpods']
for d in test_data:
    print('Test data:', d)
    inferred_vector = model.infer_vector(preprocess(d))
    sims = model.docvecs.most_similar([inferred_vector], topn=10)
    print(sims)
print('Testing time:', time.time() - start_time)

    """
    st.code(code1, language='python')

    st.write("##### 1.2. Result")
    result1 = """
Training time: 428.0623424053192
Test data: chuột logitech
[('194', 0.6743008494377136), ('845', 0.6730848550796509), ('907', 0.6221418380737305), ('1875', 0.5507464408874512), ('152', 0.5485876202583313), ('2485', 0.5318547487258911), ('795', 0.5295842289924622), ('2375', 0.528523862361908), ('609', 0.5272239446640015), ('355', 0.5214278101921082)]
Test data: loa bluetooth
[('3991', 0.6547302007675171), ('957', 0.644183337688446), ('581', 0.6272692084312439), ('306', 0.6226797699928284), ('995', 0.6158136129379272), ('415', 0.6137504577636719), ('4218', 0.6093207001686096), ('362', 0.6032481789588928), ('313', 0.6018247604370117), ('773', 0.5991814136505127)]
Test data: tai nghe apple airpods
[('581', 0.6610881686210632), ('674', 0.6213569641113281), ('299', 0.6193050742149353), ('66', 0.608256995677948), ('462', 0.6051782965660095), ('1', 0.6041195392608643), ('23', 0.6038036346435547), ('774', 0.5952202677726746), ('0', 0.5925555229187012), ('472', 0.591374933719635)]
Testing time: 428.1063139438629
    """
    st.code(result1, language='python')
    
    st.write("###### Test data")
    test_data = ['chuột logitech', 'loa bluetooth', 'tai nghe apple airpods']
    st.code(test_data, language='python')

    # Load the Doc2Vec model
    model = Doc2Vec.load('doc2vec_new.model')
    # Create an empty list to store the results
    results = []
    # Iterate over the test data and get the most similar items
    for d in test_data:
        # d = preprocess(d)
        inferred_vector = model.infer_vector(d.split())
        sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))
        n = 10
        count = 0
        for index, score in sims:
            if count < n:
                if int(index) < len(data):
                    result = {'Test data': d, 'Name': data.loc[int(index), 'name'], 'Similarity score': score}
                    results.append(result)
                count += 1
            else:
                break

    # Convert the results to a Pandas DataFrame
    df1 = pd.DataFrame(results)

    # Display the results on Streamlit
    st.write("###### Recommend for each products in dataframe format: ")
    st.write(df1)
    st.write("###### Wordcloud")

    for d in test_data:
        st.write('Test data:', d)
        # d = preprocess(d)
        inferred_vector = model.infer_vector(d.split())
        sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))
        n = 10
        count = 0
        text = ""
        for index, score in sims:
            if count < n:
                if int(index) < len(data):
                    text += data.loc[int(index), 'name'] + " "
                    st.write(data.loc[int(index), 'name'], score)
                count += 1
            else:
                break
        wordcloud = WordCloud(width=800, height=400,collocations=False, background_color="white").generate(text)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        st.pyplot()




    st.write("#### 2. Cosine Similarity")
    # load model
    cosine_new_model = Doc2Vec.load('cosine_new.model')
    st.write("##### 2.1. Coding")
    code2 = """
cosine_model = Doc2Vec(tagged_data, 
                vector_size=52, 
                window=2, 
                min_count=2, 
                workers=4, 
                epochs=200)
cosine_model.init_sims(replace=True)
# Define a function to find similar products using cosine similarity
def find_similar_products(text):       
    # Infer the vector for the input text
    inferred_vector = cosine_new_model.infer_vector(text.split())

    # Find the most similar products
    cosine_similarities = cosine_similarity([inferred_vector], cosine_new_model.docvecs.vectors)
    most_similar_indices = cosine_similarities.argsort()[0][-10:][::-1]        

    # Create a list of (product name, similarity score) tuples
    products = [(data.iloc[index]['name'], cosine_similarities[0][index]) for index in most_similar_indices]

    return products

    """
    st.code(code2, language = 'python')

    st.write("##### 2.2. Result")

    def find_similar_products(query_text):

        query_text = preprocess(query_text)

        # Infer the vector for the input text
        inferred_vector = cosine_new_model.infer_vector(query_text)

        # Find the most similar products
        cosine_similarities = cosine_similarity([inferred_vector], cosine_new_model.docvecs.vectors)
        most_similar_indices = cosine_similarities.argsort()[0][-10:][::-1]        

        # Create a list of (product name, similarity score) tuples
        products = [(data.iloc[index]['name'], cosine_similarities[0][index]) for index in most_similar_indices]

        # Add the query as a new column in the DataFrame
        products = [(query_text, product[0], product[1]) for product in products]

        return products

    # Find similar products for each query in the test_data list
    results = []
    for query in test_data:
        # st.write(f"Query: {query}")
        products = find_similar_products(query)
        temp = []
        for product in products:
            temp.append({'Query': product[0], 'Product Name': product[1], 'Similarity': product[2]})
        df = pd.DataFrame(temp)
        results.append(df)

    # Concatenate the results for all queries
    df_final = pd.concat(results)
    df_final.reset_index(inplace=True, drop=True)

    # Display the results on Streamlit
    st.write("###### Recommended products:")
    st.write(df_final)
    st.write("###### Wordcloud")
    
    for d in test_data:
        st.write('Test data:', d)
        d = preprocess(d)
        inferred_vector = cosine_new_model.infer_vector(d)
        n = 10
        sims = cosine_new_model.docvecs.most_similar([inferred_vector], topn=n)
        text = ''
        for index, score in sims:
            if int(index) < len(data):
                text += data.loc[int(index), 'name'] + ' '
                st.write(data.loc[int(index), 'name'], score)
        wordcloud = WordCloud(width=800, height=400, collocations=False, background_color='white').generate(text)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        st.pyplot()
        
    

    st.write("#### 3. Conclusion")
    st.write("")
    conclusion = """
- Although using 'cosine_similarity' time training model will be longer. But in return for more accurate results.
- Similarity of 'cosine_similarity' has a higher average value than using 'Gesim'.
- In this section, I will use cosine_model to offer recommendations for users.
"""
    st.code(conclusion, language= 'python')






elif choice == "Collaborative Filtering":
    
    st.subheader("Collaborative Filtering with PySpark")
    st.markdown("""
        <p style='text-align: justify;'> 
        - Collaborative Filtering is a technique used in recommendation systems to predict a user's interests by analyzing the behavior and preferences of similar users. It works by identifying patterns in the data and making predictions based on the similarities between users and items. Collaborative Filtering can be categorized into two types: user-based collaborative filtering and item-based collaborative filtering. In user-based collaborative filtering, the system recommends items to a user based on the items that similar users have liked. In item-based collaborative filtering, the system recommends items to a user based on the items that the user has previously liked.
        </p>
        <p style='text-align: justify;'> 
        - Pyspark is a Python library for Apache Spark, which is a powerful distributed computing framework. Pyspark provides a simple and efficient way to implement Collaborative Filtering on large datasets. To use Collaborative Filtering with Pyspark, you can use the ALS (Alternating Least Squares) algorithm. This algorithm is implemented in the ml library of Pyspark, which provides a high-level API for building recommendation systems. You can use Pyspark's ALS algorithm to train a model on a large dataset, and then use the model to make recommendations for new users.
        </p>
    """, unsafe_allow_html = True)
    
    
#     st.write("""
# - Collaborative Filtering is a technique used in recommendation systems to predict a user's interests by analyzing the behavior and preferences of similar users. It works by identifying patterns in the data and making predictions based on the similarities between users and items. Collaborative Filtering can be categorized into two types: user-based collaborative filtering and item-based collaborative filtering. In user-based collaborative filtering, the system recommends items to a user based on the items that similar users have liked. In item-based collaborative filtering, the system recommends items to a user based on the items that the user has previously liked.
# - Pyspark is a Python library for Apache Spark, which is a powerful distributed computing framework. Pyspark provides a simple and efficient way to implement Collaborative Filtering on large datasets. To use Collaborative Filtering with Pyspark, you can use the ALS (Alternating Least Squares) algorithm. This algorithm is implemented in the ml library of Pyspark, which provides a high-level API for building recommendation systems. You can use Pyspark's ALS algorithm to train a model on a large dataset, and then use the model to make recommendations for new users.
    
#     """)

    st.write("##### 1. Building the model")

    code_spark = """
import findspark
findspark.init()
from pyspark.sql import SparkSession
spark = SparkSession.builder.master("local[*]").getOrCreate()
# Check the pyspark version
import pyspark
print(pyspark.__version__)
from pyspark.ml.feature import StringIndexer
from pyspark.ml.recommendation import ALS
from pyspark.sql.functions import col, explode
import time
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator


# Use StringIndexer
product_id_index = StringIndexer(inputCol='product_id', outputCol='product_id_index')
product_model = product_id_index.fit(data)
df_indexer = product_model.transform(data)
customer_id_indexer = StringIndexer(inputCol='customer_id', outputCol='customer_id_index')
customer_model = customer_id_indexer.fit(df_indexer)
df_indexer = customer_model.transform(df_indexer)

# Cache data
df_indexer.cache()
# Split Data
(training, test) = df_indexer.randomSplit([0.8, 0.2])
# Create ALS model
als = ALS(userCol="customer_id_index", 
          itemCol="product_id_index", 
          ratingCol="rating",
          coldStartStrategy="drop", 
          nonnegative=True,
          rank=50, 
          maxIter=15, 
          regParam=0.2) 

# Train model
start_time = time.time()
model = als.fit(training)
end_time = time.time()
print(f"Time taken to train the model: {end_time - start_time:.2f} seconds")

evaluator = RegressionEvaluator(metricName="rmse", 
                                labelCol="rating",
                                predictionCol="prediction")

# Make predictions and evaluate model
predictions = model.transform(test)
rmse = evaluator.evaluate(predictions,{evaluator.labelCol: "rating"})
print("RMSE = %f" % rmse)
    """
    st.code(code_spark, language = 'python')

    st.write("###### Evaluate")
    code_spark_result = """
Time taken to train the model: 94.37 seconds
RMSE = 1.137140
    """
    st.code(code_spark_result, language = 'python')

    st.write("##### 2. Recommendations")
    st.write("###### Coding")
    code_spark2 = """
model = ALSModel.load("spark_model_collaborative_filtering")    
from pyspark.sql.functions import explode
# Get top 20 recommendations for each user
userRecs = model.recommendForAllUsers(20)
# Explode the recommendations column
userRecs = userRecs.selectExpr("customer_id_index", "explode(recommendations) as rec")
# Join with the original DataFrame to get the product ID and name
userRecs = userRecs.join(df_indexer.select("product_id_index", "product_id").distinct(),
                         userRecs.rec.product_id_index == df_indexer.product_id_index,
                         "inner").select("customer_id_index", "product_id", "rec.rating")
st.write("Top 20 recommendations for each user:")
st.write(userRecs)
data.unpersist()
#Clean up the memory from unused objects
del (
    df,
    data,
    model,
    training,
    test,
    evaluator,
    userRecs,
)
gc.collect()
    """
    st.code(code_spark2, language = 'python')
    st.write("##### Results: ")
    
    if st.button('Recommendations Results'):        
        df = pd.read_csv('Data/users_recs.csv')
        st.write("Top 20 recommendations for each user:")
        st.dataframe(df)





elif choice == "Makes Recommendations":
    st.subheader("Let's make some recommendations")

    data = pd.read_csv('Data/products_clean.csv')
    
    stop_words = []
    with open('Data/vietnamese-stopwords.txt', 'r',encoding='utf-8') as f:
        for line in f:
            stop_words.append(line.strip())

    def preprocess(text):
        words = gensim.utils.simple_preprocess(text, deacc=True)
        words = [w for w in words if w not in stop_words]
        return words


    # Load the Doc2Vec model
    cosine_new_model = gensim.models.Doc2Vec.load('cosine_new.model')

    # Create Streamlit app
    st.title('Product Recommendation')

    # Allow user input
    user_input = st.text_input('Enter a product:')

    # Generate recommendations and word cloud based on user input
    if user_input:
        preprocessed_input = preprocess(user_input)
        inferred_vector = cosine_new_model.infer_vector(preprocessed_input)
        n = 10
        sims = cosine_new_model.docvecs.most_similar([inferred_vector], topn=n)
        text = ''
        for index, score in sims:
            if int(index) < len(data):
                text += data.loc[int(index), 'name'] + ' '
                st.write(data.loc[int(index), 'name'], score)
        wordcloud = WordCloud(width=800, height=400, collocations=False, background_color='white').generate(text)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        st.pyplot()
   
