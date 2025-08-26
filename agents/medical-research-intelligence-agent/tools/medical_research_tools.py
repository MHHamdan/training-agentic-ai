"""
Medical Research Tools for MARIA
Healthcare-specific research tools and APIs
"""

import os
import requests
import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import xml.etree.ElementTree as ET

class PubMedSearchTool:
    """PubMed literature search tool"""
    
    def __init__(self):
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        # Support both environment variable names
        self.api_key = os.getenv("PUBMED_API_KEY") or os.getenv("PubMed")
        self.rate_limit_delay = 0.1  # 100ms between requests
    
    def search_literature(self, query: str, max_results: int = 20, sort: str = "relevance") -> Dict[str, Any]:
        """
        Search PubMed literature
        
        Args:
            query: Search query
            max_results: Maximum number of results
            sort: Sort order (relevance, date, etc.)
            
        Returns:
            Search results with metadata
        """
        try:
            # Construct search URL
            search_params = {
                'db': 'pubmed',
                'term': query,
                'retmax': max_results,
                'sort': sort,
                'retmode': 'json',
                'tool': 'MARIA',
                'email': 'research@example.com'
            }
            
            if self.api_key:
                search_params['api_key'] = self.api_key
            
            # Search PubMed
            search_url = f"{self.base_url}esearch.fcgi"
            response = requests.get(search_url, params=search_params, timeout=30)
            
            if response.status_code == 200:
                search_data = response.json()
                pmids = search_data.get('esearchresult', {}).get('idlist', [])
                
                if pmids:
                    # Fetch article details
                    articles = self._fetch_article_details(pmids[:max_results])
                    
                    return {
                        "success": True,
                        "query": query,
                        "total_results": len(pmids),
                        "returned_results": len(articles),
                        "articles": articles,
                        "search_timestamp": datetime.now().isoformat()
                    }
                else:
                    return {
                        "success": True,
                        "query": query,
                        "total_results": 0,
                        "returned_results": 0,
                        "articles": [],
                        "message": "No articles found for query"
                    }
            else:
                return self._create_error_response(f"PubMed search failed: {response.status_code}")
        
        except Exception as e:
            return self._create_error_response(f"PubMed search error: {str(e)}")
    
    def _fetch_article_details(self, pmids: List[str]) -> List[Dict[str, Any]]:
        """Fetch detailed article information"""
        try:
            if not pmids:
                return []
            
            # Fetch article summaries
            fetch_params = {
                'db': 'pubmed',
                'id': ','.join(pmids),
                'retmode': 'xml',
                'tool': 'MARIA',
                'email': 'research@example.com'
            }
            
            if self.api_key:
                fetch_params['api_key'] = self.api_key
            
            fetch_url = f"{self.base_url}efetch.fcgi"
            response = requests.get(fetch_url, params=fetch_params, timeout=30)
            
            if response.status_code == 200:
                return self._parse_pubmed_xml(response.text)
            else:
                return []
        
        except Exception as e:
            print(f"Error fetching article details: {e}")
            return []
    
    def _parse_pubmed_xml(self, xml_content: str) -> List[Dict[str, Any]]:
        """Parse PubMed XML response"""
        articles = []
        
        try:
            root = ET.fromstring(xml_content)
            
            for article in root.findall('.//PubmedArticle'):
                try:
                    # Extract basic information
                    pmid_elem = article.find('.//PMID')
                    pmid = pmid_elem.text if pmid_elem is not None else "Unknown"
                    
                    title_elem = article.find('.//ArticleTitle')
                    title = title_elem.text if title_elem is not None else "Title not available"
                    
                    # Extract authors
                    authors = []
                    for author in article.findall('.//Author'):
                        last_name_elem = author.find('LastName')
                        first_name_elem = author.find('ForeName')
                        if last_name_elem is not None:
                            last_name = last_name_elem.text
                            first_name = first_name_elem.text if first_name_elem is not None else ""
                            authors.append(f"{first_name} {last_name}".strip())
                    
                    # Extract journal information
                    journal_elem = article.find('.//Journal/Title')
                    journal = journal_elem.text if journal_elem is not None else "Journal not available"
                    
                    # Extract publication date
                    pub_date_elem = article.find('.//PubDate')
                    pub_date = "Date not available"
                    if pub_date_elem is not None:
                        year_elem = pub_date_elem.find('Year')
                        month_elem = pub_date_elem.find('Month')
                        if year_elem is not None:
                            year = year_elem.text
                            month = month_elem.text if month_elem is not None else "01"
                            pub_date = f"{year}-{month.zfill(2)}"
                    
                    # Extract abstract
                    abstract_elem = article.find('.//Abstract/AbstractText')
                    abstract = abstract_elem.text if abstract_elem is not None else "Abstract not available"
                    
                    articles.append({
                        "pmid": pmid,
                        "title": title,
                        "authors": authors,
                        "journal": journal,
                        "publication_date": pub_date,
                        "abstract": abstract[:500] + "..." if len(abstract) > 500 else abstract,
                        "pubmed_url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                        "confidence_score": 0.85  # Based on PubMed quality
                    })
                
                except Exception as e:
                    print(f"Error parsing individual article: {e}")
                    continue
        
        except Exception as e:
            print(f"Error parsing PubMed XML: {e}")
        
        return articles
    
    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """Create error response"""
        return {
            "success": False,
            "error": error_message,
            "articles": [],
            "total_results": 0,
            "returned_results": 0
        }


class ClinicalTrialsSearchTool:
    """ClinicalTrials.gov search tool"""
    
    def __init__(self):
        self.base_url = "https://clinicaltrials.gov/api/query/"
        self.format = "json"
    
    def search_clinical_trials(self, condition: str, max_results: int = 20) -> Dict[str, Any]:
        """
        Search clinical trials
        
        Args:
            condition: Medical condition
            max_results: Maximum number of results
            
        Returns:
            Clinical trials search results
        """
        try:
            # Construct search parameters
            params = {
                'cond': condition,
                'fmt': self.format,
                'min_rnk': 1,
                'max_rnk': max_results
            }
            
            response = requests.get(f"{self.base_url}study_fields", params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                studies = data.get('StudyFieldsResponse', {}).get('StudyFields', [])
                
                trials = []
                for study in studies:
                    try:
                        trial = {
                            "nct_id": study.get('NCTId', ['Unknown'])[0],
                            "title": study.get('BriefTitle', ['Title not available'])[0],
                            "status": study.get('OverallStatus', ['Status unknown'])[0],
                            "phase": study.get('Phase', ['Phase unknown'])[0],
                            "condition": study.get('Condition', ['Condition not specified'])[0],
                            "intervention": study.get('InterventionName', ['Intervention not specified'])[0],
                            "sponsor": study.get('LeadSponsorName', ['Sponsor not specified'])[0],
                            "start_date": study.get('StartDate', ['Date not available'])[0],
                            "completion_date": study.get('CompletionDate', ['Date not available'])[0],
                            "enrollment": study.get('EnrollmentCount', ['Not specified'])[0],
                            "url": f"https://clinicaltrials.gov/ct2/show/{study.get('NCTId', ['Unknown'])[0]}",
                            "confidence_score": 0.90  # High confidence for official clinical trials
                        }
                        trials.append(trial)
                    except Exception as e:
                        print(f"Error parsing clinical trial: {e}")
                        continue
                
                return {
                    "success": True,
                    "condition": condition,
                    "total_results": len(trials),
                    "trials": trials,
                    "search_timestamp": datetime.now().isoformat()
                }
            else:
                return self._create_trials_error_response(f"Clinical trials search failed: {response.status_code}")
        
        except Exception as e:
            return self._create_trials_error_response(f"Clinical trials search error: {str(e)}")
    
    def _create_trials_error_response(self, error_message: str) -> Dict[str, Any]:
        """Create error response for clinical trials"""
        return {
            "success": False,
            "error": error_message,
            "trials": [],
            "total_results": 0
        }


class DrugInteractionChecker:
    """Drug interaction checking tool"""
    
    def __init__(self):
        self.interactions_db = self._load_basic_interactions()
    
    def _load_basic_interactions(self) -> Dict[str, List[str]]:
        """Load basic drug interactions database"""
        # This would typically connect to a real drug interaction database
        # For now, using a basic example database
        return {
            "warfarin": ["aspirin", "ibuprofen", "acetaminophen", "amoxicillin"],
            "metformin": ["alcohol", "contrast dye", "furosemide"],
            "lisinopril": ["potassium supplements", "nsaids", "lithium"],
            "atorvastatin": ["grapefruit", "cyclosporine", "gemfibrozil"],
            "levothyroxine": ["calcium", "iron", "soy", "coffee"]
        }
    
    def check_interactions(self, medications: List[str]) -> Dict[str, Any]:
        """
        Check for drug interactions
        
        Args:
            medications: List of medications
            
        Returns:
            Drug interaction analysis
        """
        try:
            interactions_found = []
            warnings = []
            
            for i, med1 in enumerate(medications):
                med1_lower = med1.lower()
                
                # Check against interaction database
                if med1_lower in self.interactions_db:
                    interacting_drugs = self.interactions_db[med1_lower]
                    
                    for j, med2 in enumerate(medications):
                        if i != j and med2.lower() in interacting_drugs:
                            interactions_found.append({
                                "drug1": med1,
                                "drug2": med2,
                                "severity": "moderate",  # Would be determined by real database
                                "description": f"Potential interaction between {med1} and {med2}",
                                "recommendation": "Monitor closely and consult healthcare provider"
                            })
            
            # Generate warnings
            if interactions_found:
                warnings.append("Drug interactions detected - require medical review")
            
            return {
                "success": True,
                "medications_checked": medications,
                "interactions_found": len(interactions_found),
                "interactions": interactions_found,
                "warnings": warnings,
                "confidence_score": 0.75,  # Would be higher with real database
                "disclaimer": "This is a basic interaction check. Consult healthcare provider for comprehensive review.",
                "check_timestamp": datetime.now().isoformat()
            }
        
        except Exception as e:
            return {
                "success": False,
                "error": f"Drug interaction check error: {str(e)}",
                "medications_checked": medications,
                "interactions": []
            }


class MedicalGuidelinesSearchTool:
    """Medical guidelines search tool"""
    
    def __init__(self):
        self.guidelines_sources = {
            "aha": "American Heart Association",
            "acc": "American College of Cardiology", 
            "asco": "American Society of Clinical Oncology",
            "ada": "American Diabetes Association",
            "nice": "National Institute for Health and Care Excellence",
            "who": "World Health Organization"
        }
    
    def search_guidelines(self, condition: str, organization: str = "all") -> Dict[str, Any]:
        """
        Search medical guidelines
        
        Args:
            condition: Medical condition
            organization: Specific organization or "all"
            
        Returns:
            Medical guidelines search results
        """
        try:
            # This would typically search real guideline databases
            # For now, returning structured placeholder results
            
            guidelines = self._generate_sample_guidelines(condition, organization)
            
            return {
                "success": True,
                "condition": condition,
                "organization": organization,
                "guidelines_found": len(guidelines),
                "guidelines": guidelines,
                "search_timestamp": datetime.now().isoformat(),
                "disclaimer": "Sample guidelines for demonstration. Access real guidelines through official medical organizations."
            }
        
        except Exception as e:
            return {
                "success": False,
                "error": f"Guidelines search error: {str(e)}",
                "guidelines": []
            }
    
    def _generate_sample_guidelines(self, condition: str, organization: str) -> List[Dict[str, Any]]:
        """Generate sample guidelines for demonstration"""
        guidelines = []
        
        sample_guideline = {
            "title": f"{condition.title()} Management Guidelines",
            "organization": self.guidelines_sources.get(organization, "Medical Organization"),
            "publication_date": "2023-01-01",
            "summary": f"Evidence-based guidelines for {condition} diagnosis, treatment, and management.",
            "key_recommendations": [
                f"Early diagnosis and assessment of {condition}",
                f"Evidence-based treatment protocols for {condition}",
                f"Patient monitoring and follow-up guidelines",
                f"Lifestyle modifications and preventive measures"
            ],
            "evidence_level": "High",
            "confidence_score": 0.90,
            "url": f"https://example.org/guidelines/{condition.replace(' ', '-').lower()}",
            "last_updated": "2023-12-01"
        }
        
        guidelines.append(sample_guideline)
        return guidelines


def get_medical_research_tools() -> Dict[str, Any]:
    """
    Get medical research tools
    
    Returns:
        Dictionary of available medical research tools
    """
    tools = {
        "pubmed_search": PubMedSearchTool(),
        "clinical_trials_search": ClinicalTrialsSearchTool(),
        "drug_interaction_checker": DrugInteractionChecker(),
        "medical_guidelines_search": MedicalGuidelinesSearchTool()
    }
    
    # Check tool availability
    tool_status = {}
    for tool_name, tool_instance in tools.items():
        try:
            if hasattr(tool_instance, 'api_key'):
                tool_status[tool_name] = {
                    "available": bool(tool_instance.api_key),
                    "status": "API key configured" if tool_instance.api_key else "API key missing"
                }
            else:
                tool_status[tool_name] = {
                    "available": True,
                    "status": "Ready"
                }
        except Exception as e:
            tool_status[tool_name] = {
                "available": False,
                "status": f"Error: {str(e)}"
            }
    
    return {
        "tools": tools,
        "tool_status": tool_status,
        "total_tools": len(tools),
        "available_tools": sum(1 for status in tool_status.values() if status["available"])
    }